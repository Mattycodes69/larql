//! CONTINUATION-CODEC-2 W2: the frozen run, and the erratum assembly of
//! its record.
//!
//! `real_codec_2_arms_gemma3_4b` runs the frozen arms (codec_2_arms.rs)
//! resumably under four authorities. `assemble_codec_2_w2_record` is the W2
//! erratum: the as-run W2 store lost arm C's scores to a file-name
//! collision after every arm had completed (notes, W2), so the record is
//! assembled from that store read-only — every input verified against the
//! hashes taken at exit — by re-scoring C from its kept R and C logits
//! with the unchanged score(), then the unchanged pooling and rule. No
//! model arm runs in the assembly.

use std::path::Path;

use sha2::{Digest, Sha256};

use super::codec_1_c3::{file_sha256, yardstick_guard, YARDSTICK_ENCODING};
use super::codec_2::{bank, full_precision_guard, BANK_IDS_SHA256};
use super::codec_2_arms::{record, rescored_c, run_arms, strata, Ladder};
use super::codec_2_store::{w2_as_run_logits_base, Checkpoints};
use super::*;

const RECORD: &str = "codec2-gemma3-4b.json";

/// The four authorities every CODEC-2 record carries.
fn authorities(exact: (&str, &Subject), yard: (&str, &Subject)) -> Value {
    let binary = std::env::current_exe().expect("the running test binary");
    let container = |(dir, subject): (&str, &Subject)| {
        json!({
            "dir_name": Path::new(dir).file_name().map(|n| n.to_string_lossy().into_owned()),
            "index_json_sha256": file_sha256(&Path::new(dir).join("index.json")),
            "bound": format!("{:?}", subject.store.selection()),
        })
    };
    json!({
        "implementation_git_sha": git_sha(),
        "binary_sha256": file_sha256(&binary),
        "containers": {"exact": container(exact), "yardstick": container(yard)},
        "token_bank_ids_sha256": BANK_IDS_SHA256,
    })
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k}"))
}

/// CODEC-2 W2 on gemma3-4b-it: containers from LARQL_CODEC2_F32 and
/// LARQL_CODEC2_Q4K; checkpoints and the record in LARQL_CODEC2_RUN_DIR.
#[test]
#[ignore = "real containers: LARQL_CODEC2_F32 + LARQL_CODEC2_Q4K + LARQL_CODEC2_RUN_DIR (CODEC-2 W2; long, resumable)"]
fn real_codec_2_arms_gemma3_4b() {
    let _serial = serial();
    let ids = bank().expect("the frozen bank 2");
    let (f32_dir, q4k_dir) = (env("LARQL_CODEC2_F32"), env("LARQL_CODEC2_Q4K"));
    let exact = subjects::open(Path::new(&f32_dir), "gemma3-4b-it");
    let yard = subjects::open_bound(
        Path::new(&q4k_dir),
        "gemma3-4b-it.q4k",
        Some(YARDSTICK_ENCODING),
    );
    full_precision_guard(exact.store.selection()).unwrap_or_else(|stop| panic!("{stop}"));
    yardstick_guard(yard.store.selection()).unwrap_or_else(|stop| panic!("{stop}"));
    let stamp = authorities((&f32_dir, &exact), (&q4k_dir, &yard));
    let store = Checkpoints::open(Path::new(&env("LARQL_CODEC2_RUN_DIR")), &stamp);
    let backend = ProductionBackend::new();
    let (exact_ops, yard_ops) = (exact.prepare(&backend), yard.prepare(&backend));
    let ladder = Ladder::frozen();
    run_arms(
        (&exact, &exact_ops),
        (&yard, &yard_ops),
        &backend,
        &ids,
        &ladder,
        &store,
    );
    let c = strata(&store, &ladder, "c", "positions");
    let record = record("CONTINUATION-CODEC-2 W2", &stamp, &store, &ladder, &c);
    store.write(RECORD, &record);
    eprintln!(
        "wrote {}; verdict {}",
        store.path(RECORD).display(),
        record["verdict_ch"]["verdict"]
    );
}

/// Every file in `store` must match the `shasum -a 256` list taken at the
/// run's exit — no file added, missing or changed. Returns the digests.
pub(super) fn verify_inputs(store: &Checkpoints, hash_list: &str) -> Value {
    let expected: std::collections::BTreeMap<String, String> = hash_list
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (hash, path) = l.split_once("  ").expect("`<sha256>  <path>` lines");
            let name = Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            (name, hash.to_string())
        })
        .collect();
    let actual: std::collections::BTreeMap<String, String> = store
        .files()
        .into_iter()
        .map(|name| {
            let hash = format!(
                "{:x}",
                Sha256::digest(std::fs::read(store.path(&name)).unwrap())
            );
            (name, hash)
        })
        .collect();
    assert_eq!(
        actual, expected,
        "the as-run store differs from its exit hashes"
    );
    json!(actual)
}

/// W2 erratum: assemble the record from the as-run store
/// (LARQL_CODEC2_RUN_DIR, read only), verified against
/// LARQL_CODEC2_INPUT_SHA256, into LARQL_CODEC2_ASSEMBLY_DIR.
#[test]
#[ignore = "as-run W2 store: LARQL_CODEC2_RUN_DIR + LARQL_CODEC2_INPUT_SHA256 + LARQL_CODEC2_ASSEMBLY_DIR (CODEC-2 W2 erratum)"]
fn assemble_codec_2_w2_record() {
    let _serial = serial();
    let ids = bank().expect("the frozen bank 2");
    let (store, as_run) = Checkpoints::open_as_run(Path::new(&env("LARQL_CODEC2_RUN_DIR")));
    assert_eq!(
        as_run["token_bank_ids_sha256"], BANK_IDS_SHA256,
        "the run consumed bank 2"
    );
    let inputs = verify_inputs(
        &store,
        &std::fs::read_to_string(env("LARQL_CODEC2_INPUT_SHA256")).unwrap(),
    );
    let ladder = Ladder::frozen();
    let c = rescored_c(&store, &ladder, &ids, w2_as_run_logits_base);
    let mut record = record("CONTINUATION-CODEC-2 W2", &as_run, &store, &ladder, &c);
    let binary = std::env::current_exe().expect("the running test binary");
    record["assembly"] = json!({
        "why": "W2 erratum: arm C's per-position scores were overwritten by its logits index (same file name) after every arm completed; the pooling step panicked and no record was written",
        "how": "C re-scored from the as-run kept R and C logits with the unchanged score(); CH, H, NULL and residency read from the as-run scores; unchanged pooling, rule and thresholds; no model arm run",
        "assembler": {"implementation_git_sha": git_sha(), "binary_sha256": file_sha256(&binary)},
        "inputs_sha256": inputs,
        "c_seconds": "not recoverable: C's scores file carried its wall time and was overwritten",
    });
    let out = Path::new(&env("LARQL_CODEC2_ASSEMBLY_DIR")).to_path_buf();
    std::fs::create_dir_all(&out).unwrap();
    let path = out.join(RECORD);
    assert!(
        !path.exists(),
        "{} exists; an assembly is written once",
        path.display()
    );
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    eprintln!(
        "wrote {}; verdict {}",
        path.display(),
        record["verdict_ch"]["verdict"]
    );
}
