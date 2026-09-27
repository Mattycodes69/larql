//! CONTINUATION-CODEC-1 C3: the frozen arms, progressive.
//!
//! Each rung × arm is a checkpoint written atomically (temp file, then
//! rename) into LARQL_CODEC1_RUN_DIR, so the run can be stopped at any
//! point and restarted, losing only the arm in progress. R's decode
//! logits are kept per rung, so any other arm of that rung resumes
//! without recomputing R. A directory stamped with different authorities
//! (code, binary, containers, bank) is refused: a resume never mixes
//! binaries. Order per rung: R → NULL (guard) → C → H → H3; the yardstick
//! guard runs at open. The pooled rule is written once every checkpoint
//! exists. Put the directory somewhere durable — not under /tmp.

use std::path::{Path, PathBuf};

use larql_kv::CodecKvState;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::{
    summarise, PositionMetrics,
};
use serde::{de::DeserializeOwned, Serialize};

use super::codec_1_c3::{
    authorities, bank, decode_rows, null_guard, score, verdict, yardstick_guard, DECODE, RESUME,
    RUNGS, YARDSTICK_ENCODING, YARDSTICK_TOP1_MIN,
};
use super::measured::CodecResidency;
use super::*;

/// The run directory's authority stamp and the final record.
const AUTHORITIES: &str = "authorities.json";
const RECORD: &str = "codec1-c3-gemma3-4b.json";
/// The arms scored against R, in the order they run.
const SCORED_ARMS: [&str; 3] = ["c", "h", "h3"];
const F32_BYTES: usize = std::mem::size_of::<f32>();

struct Checkpoints {
    dir: PathBuf,
}

impl Checkpoints {
    /// Open `dir`, stamping it with `stamp` or refusing it if it carries
    /// another.
    fn open(dir: &Path, stamp: &Value) -> Self {
        std::fs::create_dir_all(dir).unwrap();
        let store = Self {
            dir: dir.to_path_buf(),
        };
        match store.read::<Value>(AUTHORITIES) {
            Some(held) if &held != stamp => panic!(
                "{} holds checkpoints from other authorities; a resume must not mix them. \
                 Held: {held}\nNow: {stamp}",
                dir.display()
            ),
            Some(_) => eprintln!("resuming in {}", dir.display()),
            None => store.write(AUTHORITIES, stamp),
        }
        store
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn read<T: DeserializeOwned>(&self, name: &str) -> Option<T> {
        let raw = std::fs::read_to_string(self.path(name)).ok()?;
        Some(serde_json::from_str(&raw).expect("a checkpoint written by this run"))
    }

    /// Temp file, then rename: a checkpoint is whole or absent.
    fn write_bytes(&self, name: &str, bytes: &[u8]) {
        let tmp = self.path(&format!("{name}.tmp"));
        std::fs::write(&tmp, bytes).unwrap();
        std::fs::rename(&tmp, self.path(name)).unwrap();
    }

    fn write<T: Serialize>(&self, name: &str, value: &T) {
        self.write_bytes(
            name,
            serde_json::to_string_pretty(value).unwrap().as_bytes(),
        );
    }

    /// R's decode logits for a rung: raw f32 LE, then the index that makes
    /// them count (written last, so a stop between leaves no index).
    fn write_logits(&self, name: &str, rows: &[Vec<f32>]) {
        let width = rows.first().map_or(0, Vec::len);
        let mut bytes = Vec::with_capacity(rows.len() * width * F32_BYTES);
        for x in rows.iter().flatten() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.write_bytes(&format!("{name}.f32"), &bytes);
        self.write(
            &format!("{name}.json"),
            &json!({"rows": rows.len(), "width": width}),
        );
    }

    fn read_logits(&self, name: &str) -> Option<Vec<Vec<f32>>> {
        let index: Value = self.read(&format!("{name}.json"))?;
        let (rows, width) = (
            index["rows"].as_u64()? as usize,
            index["width"].as_u64()? as usize,
        );
        let bytes = std::fs::read(self.path(&format!("{name}.f32"))).ok()?;
        assert_eq!(
            bytes.len(),
            rows * width * F32_BYTES,
            "{name}: truncated logits"
        );
        let values: Vec<f32> = bytes
            .chunks_exact(F32_BYTES)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Some(values.chunks_exact(width).map(<[f32]>::to_vec).collect())
    }
}

fn residency_json(r: Option<CodecResidency>) -> Value {
    r.map_or(Value::Null, |r| {
        json!({"holds": r.holds(), "append_born_live": r.append_born_live, "expected": r.expected,
               "strays": r.strays, "scratch_bytes": r.scratch_bytes, "scratch_bound": r.scratch_bound})
    })
}

/// C3 on gemma3-4b-it, progressive. Containers from LARQL_CODEC1_F32 and
/// LARQL_CODEC1_Q4K; checkpoints and the record in LARQL_CODEC1_RUN_DIR.
#[test]
#[ignore = "real containers: LARQL_CODEC1_F32 + LARQL_CODEC1_Q4K + LARQL_CODEC1_RUN_DIR (CODEC-1 C3; long, resumable)"]
fn real_codec_1_arms_gemma3_4b() {
    let _serial = serial();
    let ids = bank().expect("the frozen bank");
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let (f32_dir, q4k_dir) = (env("LARQL_CODEC1_F32"), env("LARQL_CODEC1_Q4K"));
    let exact = subjects::open(Path::new(&f32_dir), "gemma3-4b-it");
    let yard = subjects::open_bound(
        Path::new(&q4k_dir),
        "gemma3-4b-it.q4k",
        Some(YARDSTICK_ENCODING),
    );
    if let Err(stop) = yardstick_guard(yard.store.selection()) {
        panic!("{stop}");
    }
    let stamp = authorities((&f32_dir, &exact), (&q4k_dir, &yard));
    let store = Checkpoints::open(Path::new(&env("LARQL_CODEC1_RUN_DIR")), &stamp);
    let backend = ProductionBackend::new();
    let exact_ops = exact.prepare(&backend);
    let yard_ops = yard.prepare(&backend);

    for n in RUNGS {
        let arm_file = |arm: &str| format!("n{n}-{arm}.json");
        let null_file = format!("n{n}-null.json");
        if let Some(null) = store.read::<Value>(&null_file) {
            if null["passed"] != true {
                panic!("rung {n}: recorded NULL stop: {}", null["stop"]);
            }
        }
        let missing: Vec<&str> = SCORED_ARMS
            .into_iter()
            .filter(|a| store.read::<Value>(&arm_file(a)).is_none())
            .collect();
        if missing.is_empty() && store.read::<Value>(&null_file).is_some() {
            eprintln!("rung {n}: complete (checkpointed)");
            continue;
        }
        let start = n - DECODE;
        let journey = Journey {
            prefill: ids[..start - RESUME].to_vec(),
            resume: ids[start - RESUME..start].to_vec(),
            decode: ids[start..n].to_vec(),
        };
        let next = &ids[start + 1..=n];
        let r_name = format!("n{n}-r");
        let r = store.read_logits(&r_name).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (r, _) = decode_rows(
                &exact,
                &exact_ops,
                &backend,
                RowKvState::default(),
                &journey,
            );
            store.write_logits(&r_name, &r);
            eprintln!("rung {n}: R done in {:.0}s", t.elapsed().as_secs_f64());
            r
        });
        if store.read::<Value>(&null_file).is_none() {
            let t = std::time::Instant::now();
            let (null, _) =
                decode_rows(&exact, &exact_ops, &backend, WindowKvState::new(), &journey);
            let verdict = null_guard(&r, &null);
            store.write(
                &null_file,
                &json!({"passed": verdict.is_ok(), "stop": verdict.as_ref().err(),
                "seconds": t.elapsed().as_secs_f64()}),
            );
            if let Err(stop) = verdict {
                panic!("rung {n}: {stop}");
            }
            eprintln!("rung {n}: NULL exact in {:.0}s", t.elapsed().as_secs_f64());
        }
        for arm in missing {
            let t = std::time::Instant::now();
            let (rows, residency) = match arm {
                "c" => decode_rows(&yard, &yard_ops, &backend, RowKvState::default(), &journey),
                "h" => decode_rows(&exact, &exact_ops, &backend, CodecKvState::new(4), &journey),
                _ => decode_rows(&exact, &exact_ops, &backend, CodecKvState::new(3), &journey),
            };
            let positions = score(&r, &rows, next, n, start);
            let seconds = t.elapsed().as_secs_f64();
            store.write(
                &arm_file(arm),
                &json!({"arm": arm, "n": n, "seconds": seconds,
                "residency": residency_json(residency), "positions": positions}),
            );
            eprintln!("rung {n}: {arm} done in {seconds:.0}s");
        }
    }

    let pooled = |arm: &str| -> Vec<PositionMetrics> {
        RUNGS
            .iter()
            .flat_map(|n| {
                let v: Value = store
                    .read(&format!("n{n}-{arm}.json"))
                    .expect("every arm checkpointed");
                serde_json::from_value::<Vec<PositionMetrics>>(v["positions"].clone()).unwrap()
            })
            .collect()
    };
    let (h, h3, c) = (pooled("h"), pooled("h3"), pooled("c"));
    let (sh, sh3, sc) = (
        summarise(&h).unwrap(),
        summarise(&h3).unwrap(),
        summarise(&c).unwrap(),
    );
    let informative = sc.all.top1_agreement >= YARDSTICK_TOP1_MIN;
    let rungs: Vec<Value> = RUNGS
        .iter()
        .map(|n| {
            let arm = |a: &str| store.read::<Value>(&format!("n{n}-{a}.json")).unwrap();
            let guard = store.read::<Value>(&format!("n{n}-null.json")).unwrap();
            let (c, h, h3) = (arm("c"), arm("h"), arm("h3"));
            let seconds = json!({"null": guard.get("seconds"), "c": c.get("seconds"),
                "h": h.get("seconds"), "h3": h3.get("seconds")});
            let residency = json!({"h": h.get("residency"), "h3": h3.get("residency")});
            json!({"n": n, "null": guard, "residency": residency, "seconds": seconds})
        })
        .collect();
    let record = json!({
        "programme": "CONTINUATION-CODEC-1 C3",
        "authorities": stamp,
        "yardstick_top1": {"r_to_c": sc.all.top1_agreement, "min": YARDSTICK_TOP1_MIN, "informative": informative},
        "verdict_h": informative.then(|| verdict(&sh, &sc)),
        "verdict_h3_secondary": informative.then(|| verdict(&sh3, &sc)),
        "summaries": {"h": sh, "h3": sh3, "c": sc},
        "rungs": rungs,
    });
    store.write(RECORD, &record);
    eprintln!(
        "wrote {}; informative {informative}; verdict {}",
        store.path(RECORD).display(),
        record["verdict_h"]
    );
}

#[test]
fn checkpoints_resume_under_their_own_authorities_and_refuse_others() {
    let dir = tempfile::tempdir().unwrap();
    let stamp = json!({"implementation_git_sha": "a", "binary_sha256": "b"});
    let store = Checkpoints::open(dir.path(), &stamp);
    let rows = vec![vec![1.5f32, -0.0, f32::MIN_POSITIVE], vec![3.0, 4.0, 5.0]];
    store.write_logits("n8-r", &rows);
    let back = store.read_logits("n8-r").unwrap();
    let bits = |r: &[Vec<f32>]| r.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&back), bits(&rows), "logits round-trip bit for bit");
    store.write("n8-h.json", &json!({"arm": "h"}));
    assert!(
        !store.path("n8-h.json.tmp").exists(),
        "a write is whole or absent"
    );

    let resumed = Checkpoints::open(dir.path(), &stamp);
    assert_eq!(resumed.read::<Value>("n8-h.json").unwrap()["arm"], "h");

    let other = json!({"implementation_git_sha": "a", "binary_sha256": "c"});
    let refused = std::panic::catch_unwind(|| Checkpoints::open(dir.path(), &other));
    assert!(
        refused.is_err(),
        "another binary's checkpoints must not be resumed"
    );
}
