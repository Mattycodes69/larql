//! CONTINUATION-CODEC-2 W2's frozen arms, progressive and resumable.
//!
//! Each rung × arm is a checkpoint written atomically into
//! LARQL_CODEC2_RUN_DIR; a directory stamped with other authorities (code,
//! binary, containers, bank) is refused, so a resume never mixes binaries.
//! Order per rung: R (logits kept) → NULL (guard) → C (logits kept) → CH
//! → H. Guards stop the run UNINFORMATIVE; the pooled verdict is written
//! once every checkpoint exists. The checkpoint store restates CODEC-1's
//! (private to its frozen module, which is not edited — notes W1).

use std::path::{Path, PathBuf};

use larql_kv::CodecKvState;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::{
    summarise, PositionMetrics,
};
use serde::{de::DeserializeOwned, Serialize};

use super::codec_1_c3::{
    decode_rows, file_sha256, null_guard, score, yardstick_guard, YARDSTICK_ENCODING,
};
use super::codec_2::{
    adjudicate, bank, full_precision_guard, BANK_IDS_SHA256, DECODE, RESUME, RUNGS,
};
use super::measured::CodecResidency;
use super::*;

const AUTHORITIES: &str = "authorities.json";
const RECORD: &str = "codec2-gemma3-4b.json";
/// The KV bits of CH and H, as frozen.
const CODEC_BITS: u8 = 4;
const F32_BYTES: usize = std::mem::size_of::<f32>();

struct Checkpoints {
    dir: PathBuf,
}

impl Checkpoints {
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

/// CODEC-1 established codec residency; here a miss is a provider or
/// instrument fault that stops the run.
fn residency_regression(arm: &str, n: usize, r: Option<CodecResidency>) -> Value {
    let r = r.unwrap_or_else(|| panic!("rung {n}: {arm} reported no codec residency"));
    assert!(
        r.holds(),
        "rung {n}: {arm} residency regression — live {} expected {} strays {} scratch {}/{}",
        r.append_born_live,
        r.expected,
        r.strays,
        r.scratch_bytes,
        r.scratch_bound
    );
    json!({"holds": true, "append_born_live": r.append_born_live, "expected": r.expected,
           "strays": r.strays, "scratch_bytes": r.scratch_bytes, "scratch_bound": r.scratch_bound})
}

/// Borrow each rung's positions as one stratum.
fn refs(v: &[Vec<PositionMetrics>]) -> Vec<&[PositionMetrics]> {
    v.iter().map(Vec::as_slice).collect()
}

/// CODEC-2 W2 on gemma3-4b-it: containers from LARQL_CODEC2_F32 and
/// LARQL_CODEC2_Q4K; checkpoints and the record in LARQL_CODEC2_RUN_DIR.
#[test]
#[ignore = "real containers: LARQL_CODEC2_F32 + LARQL_CODEC2_Q4K + LARQL_CODEC2_RUN_DIR (CODEC-2 W2; long, resumable)"]
fn real_codec_2_arms_gemma3_4b() {
    let _serial = serial();
    let ids = bank().expect("the frozen bank 2");
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
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
    let exact_ops = exact.prepare(&backend);
    let yard_ops = yard.prepare(&backend);

    for n in RUNGS {
        let file = |arm: &str| format!("n{n}-{arm}.json");
        let start = n - DECODE;
        let journey = Journey {
            prefill: ids[..start - RESUME].to_vec(),
            resume: ids[start - RESUME..start].to_vec(),
            decode: ids[start..n].to_vec(),
        };
        let next = &ids[start + 1..=n];
        let timed = |arm: &str, t: std::time::Instant| {
            eprintln!("rung {n}: {arm} done in {:.0}s", t.elapsed().as_secs_f64());
            t.elapsed().as_secs_f64()
        };

        let r = store.read_logits(&format!("n{n}-r")).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (r, _) = decode_rows(
                &exact,
                &exact_ops,
                &backend,
                RowKvState::default(),
                &journey,
            );
            store.write_logits(&format!("n{n}-r"), &r);
            timed("R", t);
            r
        });
        match store.read::<Value>(&file("null")) {
            Some(null) if null["passed"] != true => {
                panic!("rung {n}: recorded NULL stop: {}", null["stop"])
            }
            Some(_) => {}
            None => {
                let t = std::time::Instant::now();
                let (null, _) =
                    decode_rows(&exact, &exact_ops, &backend, WindowKvState::new(), &journey);
                let verdict = null_guard(&r, &null);
                store.write(
                    &file("null"),
                    &json!({"passed": verdict.is_ok(),
                    "stop": verdict.as_ref().err(), "seconds": timed("NULL", t)}),
                );
                if let Err(stop) = verdict {
                    panic!("rung {n}: {stop}");
                }
            }
        }
        let c = store.read_logits(&format!("n{n}-c")).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (c, _) = decode_rows(&yard, &yard_ops, &backend, RowKvState::default(), &journey);
            store.write(
                &file("c"),
                &json!({"arm": "c", "n": n, "seconds": timed("C", t),
                "positions": score(&r, &c, next, n, start)}),
            );
            store.write_logits(&format!("n{n}-c"), &c);
            c
        });
        if store.read::<Value>(&file("ch")).is_none() {
            let t = std::time::Instant::now();
            let (ch, res) = decode_rows(
                &yard,
                &yard_ops,
                &backend,
                CodecKvState::new(CODEC_BITS),
                &journey,
            );
            let residency = residency_regression("CH", n, res);
            store.write(
                &file("ch"),
                &json!({"arm": "ch", "n": n, "seconds": timed("CH", t),
                "residency": residency, "positions": score(&r, &ch, next, n, start),
                "positions_vs_c": score(&c, &ch, next, n, start)}),
            );
        }
        if store.read::<Value>(&file("h")).is_none() {
            let t = std::time::Instant::now();
            let (h, res) = decode_rows(
                &exact,
                &exact_ops,
                &backend,
                CodecKvState::new(CODEC_BITS),
                &journey,
            );
            let residency = residency_regression("H", n, res);
            store.write(
                &file("h"),
                &json!({"arm": "h", "n": n, "seconds": timed("H", t),
                "residency": residency, "positions": score(&r, &h, next, n, start)}),
            );
        }
    }

    let strata = |arm: &str, key: &str| -> Vec<Vec<PositionMetrics>> {
        RUNGS
            .iter()
            .map(|n| {
                let v: Value = store
                    .read(&format!("n{n}-{arm}.json"))
                    .expect("every arm checkpointed");
                serde_json::from_value(v[key].clone()).unwrap()
            })
            .collect()
    };
    let (c, ch, h, c_ch) = (
        strata("c", "positions"),
        strata("ch", "positions"),
        strata("h", "positions"),
        strata("ch", "positions_vs_c"),
    );
    let summary = |v: &[Vec<PositionMetrics>]| summarise(&v.concat()).unwrap();
    let rungs: Vec<Value> = RUNGS
        .iter()
        .map(|n| {
            let arm = |a: &str| store.read::<Value>(&format!("n{n}-{a}.json")).unwrap();
            let (ch, h) = (arm("ch"), arm("h"));
            json!({"n": n, "null": arm("null"),
                "residency": {"ch": ch.get("residency"), "h": h.get("residency")},
                "seconds": {"c": arm("c").get("seconds"), "ch": ch.get("seconds"), "h": h.get("seconds")}})
        })
        .collect();
    let record = json!({
        "programme": "CONTINUATION-CODEC-2 W2",
        "authorities": stamp,
        "verdict_ch": adjudicate(&refs(&c), &refs(&ch)),
        "reported": {
            "r_c": summary(&c), "r_ch_end_to_end": summary(&ch), "r_h_isolated_codec": summary(&h),
            "c_ch_kl": summary(&c_ch),
        },
        "rungs": rungs,
    });
    store.write(RECORD, &record);
    eprintln!(
        "wrote {}; verdict {}",
        store.path(RECORD).display(),
        record["verdict_ch"]["verdict"]
    );
}
