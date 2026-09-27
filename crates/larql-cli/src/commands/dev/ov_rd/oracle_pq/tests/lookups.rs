//! The missing-entry errors of the per-head and per-`(head, config)`
//! lookups, and an eval prompt that encodes to no tokens.

use std::collections::HashMap;

use crate::commands::dev::ov_rd::types::{HeadId, PqConfig};

use super::super::probe::{lookup, lookup_head, HeadConfigMap};
use super::fixture::PROMPTS;
use super::run_with;

const HEAD: HeadId = HeadId { layer: 1, head: 0 };
const CONFIG: PqConfig = PqConfig {
    k: 8,
    groups: 2,
    bits_per_group: 4,
};

#[test]
fn a_present_entry_is_returned() {
    let per_head = HashMap::from([(HEAD, 7usize)]);
    let per_point: HeadConfigMap<usize> = HashMap::from([((HEAD, CONFIG), 9usize)]);
    assert_eq!(*lookup_head(&per_head, HEAD, "basis for").unwrap(), 7);
    assert_eq!(*lookup(&per_point, HEAD, CONFIG, "codes for").unwrap(), 9);
}

#[test]
fn a_missing_entry_names_the_head_and_config() {
    let per_head: HashMap<HeadId, usize> = HashMap::new();
    let per_point: HeadConfigMap<usize> = HashMap::new();
    let err = lookup_head(&per_head, HEAD, "basis for").unwrap_err();
    assert_eq!(err.to_string(), "missing basis for L1 H0");
    let err = lookup(&per_point, HEAD, CONFIG, "Mode D table for LSH group probe").unwrap_err();
    assert_eq!(
        err.to_string(),
        "missing Mode D table for LSH group probe L1 H0 PqConfig { k: 8, groups: 2, bits_per_group: 4 }"
    );
}

#[test]
fn an_eval_prompt_with_no_tokens_is_skipped() {
    let root = tempfile::tempdir().unwrap();
    let prompts = root.path().join("prompts.jsonl");
    let mut lines = PROMPTS
        .iter()
        .map(|(id, stratum, prompt)| {
            serde_json::json!({ "id": id, "stratum": stratum, "prompt": prompt }).to_string()
        })
        .collect::<Vec<_>>();
    lines.push(serde_json::json!({ "id": "empty", "prompt": "" }).to_string());
    std::fs::write(&prompts, lines.join("\n")).unwrap();
    let out = root.path().join("out");
    let prompts = prompts.display().to_string();
    run_with(&out, &["--prompts", &prompts, "--pq-iters", "2"]).unwrap();

    let text = std::fs::read_to_string(out.join("oracle_pq.json")).unwrap();
    let report: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(report["eval_prompts_seen"], PROMPTS.len() + 1);
    let per_prompt = report["heads"][0]["points"][0]["per_prompt"]
        .as_array()
        .unwrap();
    assert_eq!(per_prompt.len(), PROMPTS.len());
}
