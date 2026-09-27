use super::*;

#[test]
fn parse_shard_filename_canonical_layout() {
    let p = std::path::PathBuf::from("/x/Kimi-K2.6-UD-Q8_K_XL-00003-of-00014.gguf");
    let (prefix, idx, total) = parse_shard_filename(&p).unwrap();
    assert_eq!(prefix, "Kimi-K2.6-UD-Q8_K_XL");
    assert_eq!(idx, 2);
    assert_eq!(total, 14);
}

#[test]
fn parse_shard_filename_rejects_single_file() {
    let p = std::path::PathBuf::from("/x/llama-3.1-8b-q4.gguf");
    assert!(parse_shard_filename(&p).is_none());
}

#[test]
fn parse_shard_filename_rejects_unmatched_widths() {
    let p = std::path::PathBuf::from("/x/foo-00003-of-0014.gguf");
    assert!(parse_shard_filename(&p).is_none());
}

#[test]
fn parse_shard_filename_supports_3digit_split() {
    let p = std::path::PathBuf::from("/x/foo-001-of-003.gguf");
    let (prefix, idx, total) = parse_shard_filename(&p).unwrap();
    assert_eq!(prefix, "foo");
    assert_eq!(idx, 0);
    assert_eq!(total, 3);
}

#[test]
fn parse_shard_filename_rejects_index_zero() {
    let p = std::path::PathBuf::from("/x/foo-00000-of-00003.gguf");
    assert!(parse_shard_filename(&p).is_none());
}

#[test]
fn parse_shard_filename_rejects_index_exceeding_total() {
    let p = std::path::PathBuf::from("/x/foo-00004-of-00003.gguf");
    assert!(parse_shard_filename(&p).is_none());
}

#[test]
fn parse_shard_filename_rejects_no_trailing_digits() {
    let p = std::path::PathBuf::from("/x/foo-abc.gguf");
    assert!(parse_shard_filename(&p).is_none());
}

#[test]
fn parse_shard_filename_rejects_all_digits_before_of() {
    // "00003-of-00003.gguf" — digits run to start, no prefix with '-'
    let p = std::path::PathBuf::from("/x/00003-of-00003.gguf");
    assert!(parse_shard_filename(&p).is_none());
}

#[test]
fn discover_shard_siblings_rejects_total_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    for i in 1..=3 {
        std::fs::File::create(dir.path().join(format!("m-{i:0>5}-of-00003.gguf"))).unwrap();
    }
    let first = dir.path().join("m-00001-of-00003.gguf");
    let err = discover_shard_siblings(dir.path(), &first, 5).unwrap_err();
    assert!(
        format!("{err}").contains("shard total mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn discover_shard_siblings_finds_all_in_order() {
    let dir = tempfile::tempdir().unwrap();
    for i in 1..=3 {
        std::fs::File::create(dir.path().join(format!("model-{i:0>5}-of-00003.gguf"))).unwrap();
    }
    let middle = dir.path().join("model-00002-of-00003.gguf");
    let paths = discover_shard_siblings(dir.path(), &middle, 3).unwrap();
    assert_eq!(paths.len(), 3);
    assert!(paths[0].ends_with("model-00001-of-00003.gguf"));
    assert!(paths[1].ends_with("model-00002-of-00003.gguf"));
    assert!(paths[2].ends_with("model-00003-of-00003.gguf"));
}

#[test]
fn discover_shard_siblings_finds_3digit_splits() {
    let dir = tempfile::tempdir().unwrap();
    for i in 1..=3 {
        std::fs::File::create(dir.path().join(format!("foo-{i:0>3}-of-003.gguf"))).unwrap();
    }
    let first = dir.path().join("foo-001-of-003.gguf");
    let paths = discover_shard_siblings(dir.path(), &first, 3).unwrap();
    assert_eq!(paths.len(), 3);
    assert!(paths[0].ends_with("foo-001-of-003.gguf"));
    assert!(paths[1].ends_with("foo-002-of-003.gguf"));
    assert!(paths[2].ends_with("foo-003-of-003.gguf"));
}

#[test]
fn discover_shard_siblings_errors_when_one_missing() {
    let dir = tempfile::tempdir().unwrap();
    for i in [1usize, 3] {
        std::fs::File::create(dir.path().join(format!("m-{i:0>5}-of-00003.gguf"))).unwrap();
    }
    let first = dir.path().join("m-00001-of-00003.gguf");
    let err = discover_shard_siblings(dir.path(), &first, 3).unwrap_err();
    assert!(
        format!("{err}").contains("missing expected sibling"),
        "unexpected error: {err}"
    );
}

/// End-to-end multi-shard open: two real GGUF files with different
/// tensors in each, joined via canonical `-NNNNN-of-00002.gguf` layout.
/// Verifies discovery, shard_idx assignment, and per-shard tensor
/// reads via `load_tensors`.
#[test]
fn open_multi_shard_combines_tensors_from_all_shards() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();

    let write_shard =
        |idx: usize, tensor_ids: &[usize], metas: &[(&str, u32)]| -> std::path::PathBuf {
            let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&3u32.to_le_bytes()).unwrap();
            file.write_all(&(tensor_ids.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();

            for (k, v) in metas {
                let kb = k.as_bytes();
                file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(kb).unwrap();
                file.write_all(&4u32.to_le_bytes()).unwrap(); // u32 type tag
                file.write_all(&v.to_le_bytes()).unwrap();
            }

            for (rel, &tid) in tensor_ids.iter().enumerate() {
                let name = format!("blk.{tid}.ffn_down.weight");
                let nb = name.as_bytes();
                file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(nb).unwrap();
                file.write_all(&2u32.to_le_bytes()).unwrap();
                file.write_all(&2u64.to_le_bytes()).unwrap();
                file.write_all(&2u64.to_le_bytes()).unwrap();
                file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                    .unwrap();
                let off = (rel as u64) * 16;
                file.write_all(&off.to_le_bytes()).unwrap();
            }

            let pos = file.stream_position().unwrap();
            let aligned = pos.div_ceil(32) * 32;
            file.write_all(&vec![0u8; (aligned - pos) as usize])
                .unwrap();

            for &tid in tensor_ids {
                for off in 0..4 {
                    file.write_all(&((tid as f32) + 0.1 * off as f32).to_le_bytes())
                        .unwrap();
                }
            }
            file.flush().unwrap();
            path
        };

    let p1 = write_shard(
        1,
        &[0, 1],
        &[
            ("split.no", 0),
            ("split.count", 2),
            ("split.tensors.count", 4),
        ],
    );
    let _p2 = write_shard(
        2,
        &[2, 3],
        &[
            ("split.no", 1),
            ("split.count", 2),
            ("split.tensors.count", 4),
        ],
    );

    let gguf = GgufFile::open(&p1).unwrap();
    assert_eq!(gguf.shards.len(), 2);
    assert_eq!(gguf.tensor_infos.len(), 4);
    for (i, info) in gguf.tensor_infos.iter().enumerate() {
        let expected = if i < 2 { 0 } else { 1 };
        assert_eq!(
            info.shard_idx, expected,
            "tensor {i} ({}) shard mismatch",
            info.name
        );
    }

    let (tensors, _) = gguf.load_tensors().unwrap();
    assert_eq!(tensors.len(), 4);
    for tid in 0..4 {
        let key = format!("layers.{tid}.mlp.down_proj.weight");
        let arr = tensors.get(&key).unwrap_or_else(|| panic!("missing {key}"));
        assert!(
            (arr[[0, 0]] - tid as f32).abs() < 1e-6,
            "tensor {tid} top-left {} != {tid}",
            arr[[0, 0]]
        );
    }
}

#[test]
fn open_rejects_multi_shard_when_a_shard_file_is_missing() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("m-00001-of-00002.gguf");
    let mut file = std::fs::File::create(&p).unwrap();
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    let k = "split.count".as_bytes();
    file.write_all(&(k.len() as u64).to_le_bytes()).unwrap();
    file.write_all(k).unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.flush().unwrap();

    let err = match GgufFile::open(&p) {
        Ok(_) => panic!("expected error for missing sibling shard"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("missing expected sibling"),
        "unexpected error: {err}"
    );
}

#[test]
fn open_multi_shard_via_non_first_shard() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let write_shard = |idx: usize, tensor_ids: &[usize], metas: &[(&str, u32)]| {
        let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(tensor_ids.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();
        for (k, v) in metas {
            let kb = k.as_bytes();
            file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
            file.write_all(kb).unwrap();
            file.write_all(&4u32.to_le_bytes()).unwrap();
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        for (rel, &tid) in tensor_ids.iter().enumerate() {
            let name = format!("blk.{tid}.ffn_down.weight");
            let nb = name.as_bytes();
            file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
            file.write_all(nb).unwrap();
            file.write_all(&2u32.to_le_bytes()).unwrap();
            file.write_all(&2u64.to_le_bytes()).unwrap();
            file.write_all(&2u64.to_le_bytes()).unwrap();
            file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                .unwrap();
            let off = (rel as u64) * 16;
            file.write_all(&off.to_le_bytes()).unwrap();
        }
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for &tid in tensor_ids {
            for off in 0..4 {
                file.write_all(&((tid as f32) + 0.1 * off as f32).to_le_bytes())
                    .unwrap();
            }
        }
        file.flush().unwrap();
        path
    };

    let _p1 = write_shard(1, &[0], &[("split.count", 2), ("split.tensors.count", 2)]);
    let p2 = write_shard(2, &[1], &[("split.count", 2), ("split.tensors.count", 2)]);

    let gguf = GgufFile::open(&p2).unwrap();
    assert_eq!(gguf.shards.len(), 2);
    assert_eq!(gguf.tensor_infos.len(), 2);
}

#[test]
fn open_multi_shard_discovers_via_filename_when_split_count_absent() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let write_shard = |idx: usize, n_tensors: usize, metas: &[(&str, u32)]| {
        let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(n_tensors as u64).to_le_bytes()).unwrap();
        file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();
        for (k, v) in metas {
            let kb = k.as_bytes();
            file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
            file.write_all(kb).unwrap();
            file.write_all(&4u32.to_le_bytes()).unwrap();
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        for i in 0..n_tensors {
            let name = format!("blk.{i}.ffn_down.weight");
            let nb = name.as_bytes();
            file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
            file.write_all(nb).unwrap();
            file.write_all(&2u32.to_le_bytes()).unwrap();
            file.write_all(&1u64.to_le_bytes()).unwrap();
            file.write_all(&1u64.to_le_bytes()).unwrap();
            file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                .unwrap();
            file.write_all(&((i as u64) * 4).to_le_bytes()).unwrap();
        }
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for i in 0..n_tensors {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        path
    };

    // No split.count metadata — open must detect via filename pattern
    let p1 = write_shard(1, 1, &[]);
    let _p2 = write_shard(2, 1, &[]);

    let gguf = GgufFile::open(&p1).unwrap();
    assert_eq!(gguf.shards.len(), 2);
    assert_eq!(gguf.tensor_infos.len(), 2);
}

#[test]
fn open_multi_shard_rejects_tensor_count_mismatch() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let write_shard = |idx: usize, n_tensors: usize, metas: &[(&str, u32)]| {
        let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(n_tensors as u64).to_le_bytes()).unwrap();
        file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();
        for (k, v) in metas {
            let kb = k.as_bytes();
            file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
            file.write_all(kb).unwrap();
            file.write_all(&4u32.to_le_bytes()).unwrap();
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        for i in 0..n_tensors {
            let name = format!("blk.{i}.ffn_down.weight");
            let nb = name.as_bytes();
            file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
            file.write_all(nb).unwrap();
            file.write_all(&2u32.to_le_bytes()).unwrap();
            file.write_all(&1u64.to_le_bytes()).unwrap();
            file.write_all(&1u64.to_le_bytes()).unwrap();
            file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                .unwrap();
            file.write_all(&((i as u64) * 4).to_le_bytes()).unwrap();
        }
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for i in 0..n_tensors {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        path
    };

    // split.tensors.count says 99 but actual is 2
    let p1 = write_shard(1, 1, &[("split.count", 2), ("split.tensors.count", 99)]);
    let _p2 = write_shard(2, 1, &[("split.count", 2), ("split.tensors.count", 99)]);

    let err = match GgufFile::open(&p1) {
        Ok(_) => panic!("expected error for tensor count mismatch"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("tensor count mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn multi_shard_tensor_info_accessors() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m-00001-of-00001.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap(); // 1 tensor
    file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
    let name = b"blk.0.ffn_down.weight";
    file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    file.write_all(name).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
    file.write_all(&3u64.to_le_bytes()).unwrap(); // dim0
    file.write_all(&4u64.to_le_bytes()).unwrap(); // dim1
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap(); // offset
    let pos = file.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    file.write_all(&vec![0u8; (aligned - pos) as usize])
        .unwrap();
    file.write_all(&[0u8; 3 * 4 * 4]).unwrap(); // 3x4 f32
    file.flush().unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    let info = &gguf.tensor_infos[0];
    assert_eq!(info.name(), "blk.0.ffn_down.weight");
    assert_eq!(info.n_dims(), 2);
    assert_eq!(info.dims(), &[3, 4]);
    assert_eq!(info.tensor_type(), crate::quant::ggml::TYPE_F32);
    assert_eq!(info.offset(), 0);
    assert_eq!(info.shard_idx(), 0);
}

#[test]
fn open_single_rejects_bad_magic() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
    file.flush().unwrap();

    let err = match GgufFile::open(&path) {
        Ok(_) => panic!("expected error for bad magic"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("not a GGUF file"), "{err}");
}

#[test]
fn open_single_rejects_unsupported_version() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v99.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&99u32.to_le_bytes()).unwrap();
    file.flush().unwrap();

    let err = match GgufFile::open(&path) {
        Ok(_) => panic!("expected error for version 99"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("unsupported GGUF version"),
        "{err}"
    );
}
