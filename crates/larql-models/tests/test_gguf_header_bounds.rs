//! A GGUF header's counts and lengths are untrusted: each must be refused
//! when the file cannot hold it, instead of sizing an allocation from it.

use larql_models::loading::gguf::{GgufFile, GgufValue, GgufWriter};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const ALIGNMENT_KEY: &str = "general.alignment";
/// Enough slack for one metadata entry or tensor info to pass its count check.
const PADDING: usize = 64;

/// Hand-assembled GGUF header bytes, so tests can state any count they like.
#[derive(Default)]
struct Header {
    bytes: Vec<u8>,
}

impl Header {
    fn new(n_tensors: u64, n_metadata: u64) -> Self {
        let mut h = Self::default();
        h.bytes.extend_from_slice(GGUF_MAGIC);
        h.u32(GGUF_VERSION).u64(n_tensors).u64(n_metadata);
        h
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn string(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.bytes.extend_from_slice(s.as_bytes());
        self
    }
    /// Trailing bytes so the header's own counts pass their budget check and
    /// the check under test is the one that fires.
    fn pad(&mut self, n: usize) -> &mut Self {
        self.bytes.resize(self.bytes.len() + n, 0);
        self
    }
    fn open(&self) -> Result<GgufFile, String> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crafted.gguf");
        std::fs::write(&path, &self.bytes).unwrap();
        GgufFile::open(&path).map_err(|e| e.to_string())
    }
}

fn assert_refused(result: Result<GgufFile, String>, what: &str) {
    let err = result.err().expect("crafted header must be refused");
    assert!(err.contains(what), "expected `{what}` in error, got: {err}");
}

#[test]
fn metadata_count_beyond_file_is_refused() {
    assert_refused(Header::new(0, u64::MAX).open(), "metadata count");
}

#[test]
fn tensor_count_beyond_file_is_refused() {
    assert_refused(Header::new(u64::MAX, 0).open(), "tensor count");
}

#[test]
fn string_length_beyond_file_is_refused() {
    let mut h = Header::new(0, 1);
    h.u64(u64::MAX).pad(PADDING); // key length
    assert_refused(h.open(), "string length");
}

#[test]
fn array_length_beyond_file_is_refused() {
    let mut h = Header::new(0, 1);
    h.string("k")
        .u32(GGUF_TYPE_ARRAY)
        .u32(GGUF_TYPE_UINT32)
        .u64(u64::MAX);
    assert_refused(h.open(), "array length");
}

#[test]
fn string_array_uses_length_prefix_as_minimum_element() {
    // 2 string elements need at least 16 bytes of length prefixes; give 8.
    let mut h = Header::new(0, 1);
    h.string("k")
        .u32(GGUF_TYPE_ARRAY)
        .u32(GGUF_TYPE_STRING)
        .u64(2)
        .u64(0);
    assert_refused(h.open(), "array length");
}

#[test]
fn tensor_dim_count_beyond_file_is_refused() {
    let mut h = Header::new(1, 0);
    h.string("t").u32(u32::MAX).pad(PADDING);
    assert_refused(h.open(), "tensor dims");
}

#[test]
fn non_power_of_two_alignment_is_refused() {
    let mut h = Header::new(0, 1);
    h.string(ALIGNMENT_KEY).u32(GGUF_TYPE_UINT32).u32(48);
    assert_refused(h.open(), "power of two");
}

#[test]
fn zero_alignment_is_refused() {
    let mut h = Header::new(0, 1);
    h.string(ALIGNMENT_KEY).u32(GGUF_TYPE_UINT32).u32(0);
    assert_refused(h.open(), "power of two");
}

#[test]
fn declared_alignment_positions_the_data_section() {
    const DECLARED: u32 = 256;
    let mut h = Header::new(0, 1);
    h.string(ALIGNMENT_KEY).u32(GGUF_TYPE_UINT32).u32(DECLARED);
    let header_len = h.bytes.len() as u64;
    let gguf = h.open().unwrap();
    let declared = u64::from(DECLARED);
    assert_eq!(gguf.data_offset, header_len.div_ceil(declared) * declared);
    // The header is shorter than 256 bytes, so the 32-byte default would
    // land elsewhere: this distinguishes "declared" from "defaulted".
    assert!(header_len < declared);
    assert_eq!(gguf.data_offset, declared);
}

#[test]
fn absent_alignment_defaults_to_32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.gguf");
    let mut w = GgufWriter::new();
    w.meta("general.name", GgufValue::String("x".into()));
    w.write_to_file(&path).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(gguf.data_offset % 32, 0);
}

/// A GGUF written by the crate's own writer with `meta` added.
fn written(meta: Vec<(&str, GgufValue)>) -> Result<GgufFile, String> {
    let mut w = GgufWriter::new();
    w.meta("general.architecture", GgufValue::String("llama".into()));
    for (k, v) in meta {
        w.meta(k, v);
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written.gguf");
    w.write_to_file(&path).unwrap();
    GgufFile::open(&path).map_err(|e| e.to_string())
}

#[test]
fn a_non_integer_alignment_is_refused() {
    assert_refused(
        written(vec![(ALIGNMENT_KEY, GgufValue::String("32".into()))]),
        "must be an integer",
    );
}

#[test]
fn a_declared_rms_epsilon_reaches_the_config() {
    let gguf = written(vec![(
        "llama.attention.layer_norm_rms_epsilon",
        GgufValue::F32(1e-5),
    )])
    .unwrap();
    let eps = gguf.to_config_json()["rms_norm_eps"].as_f64().unwrap();
    assert!((eps - 1e-5).abs() < 1e-9, "{eps}");
}

/// One-tensor file whose tensor declares `dims` at data `offset`.
fn one_tensor(dims: &[u64], offset: u64) -> (tempfile::TempDir, std::path::PathBuf) {
    let mut h = Header::new(1, 1);
    h.string("general.architecture")
        .u32(GGUF_TYPE_STRING)
        .string("llama");
    h.string("token_embd.weight").u32(dims.len() as u32);
    for &d in dims {
        h.u64(d);
    }
    h.u32(0).u64(offset).pad(PADDING);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.gguf");
    std::fs::write(&path, &h.bytes).unwrap();
    (dir, path)
}

fn load_err(path: &std::path::Path) -> String {
    match larql_models::loading::gguf::load_gguf(path) {
        Ok(_) => panic!("crafted tensor table must be refused"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_tensor_offset_past_u64_is_refused_at_load() {
    let (_dir, path) = one_tensor(&[4], u64::MAX);
    let err = load_err(&path);
    assert!(err.contains("overflows"), "{err}");
}

#[test]
fn a_tensor_element_count_past_usize_is_refused_at_load() {
    let (_dir, path) = one_tensor(&[u64::MAX, 4], 0);
    let err = load_err(&path);
    assert!(err.contains("overflows"), "{err}");
}
