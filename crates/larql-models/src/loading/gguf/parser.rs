//! GGUF file parsing — GgufFile::open/open_single, shard filename parsing, sibling discovery.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use crate::detect::ModelError;

use super::bounded::{checked_count, Budgeted};
use super::constants::{GGUF_DEFAULT_ALIGNMENT, GGUF_GENERAL_ALIGNMENT, GGUF_MAGIC};
use super::reader::{read_string, read_u32, read_u64, read_value, GGUF_STRING_LEN_BYTES};
use super::types::GgufValue;

/// Width of a fixed-size `u32` header field.
const U32_BYTES: u64 = std::mem::size_of::<u32>() as u64;
/// Width of a fixed-size `u64` header field.
const U64_BYTES: u64 = std::mem::size_of::<u64>() as u64;
/// Smallest metadata entry: key length prefix + value type tag + a 1-byte value.
const MIN_METADATA_ENTRY_BYTES: u64 = GGUF_STRING_LEN_BYTES + U32_BYTES + 1;
/// Smallest tensor info: name length prefix + `n_dims` + type + offset.
const MIN_TENSOR_INFO_BYTES: u64 = GGUF_STRING_LEN_BYTES + U32_BYTES + U32_BYTES + U64_BYTES;

/// The data-section alignment the file declares, or GGUF's default of 32.
/// A declared alignment must be a non-zero power of two.
fn data_alignment(metadata: &HashMap<String, GgufValue>) -> Result<u64, ModelError> {
    let Some(value) = metadata.get(GGUF_GENERAL_ALIGNMENT) else {
        return Ok(GGUF_DEFAULT_ALIGNMENT);
    };
    let alignment = value.as_u32().ok_or_else(|| {
        ModelError::Parse(format!(
            "{GGUF_GENERAL_ALIGNMENT} must be an integer, got {value:?}"
        ))
    })?;
    if !alignment.is_power_of_two() {
        return Err(ModelError::Parse(format!(
            "{GGUF_GENERAL_ALIGNMENT} must be a non-zero power of two, got {alignment}"
        )));
    }
    Ok(u64::from(alignment))
}
use super::types::{GgufFile, GgufTensorInfo, ShardInfo};

/// Parse a multi-shard GGUF filename of the form
/// `<prefix>-<NNNNN>-of-<NNNNN>.gguf` (canonical llama.cpp split layout)
/// and return `(prefix_without_dashes, this_shard_idx_0based, total_shards)`.
///
/// Returns `None` for filenames that don't match the pattern (i.e. single
/// files); the caller treats those as single-shard GGUFs.
pub(crate) fn parse_shard_filename(path: &Path) -> Option<(String, usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    // Tail must be `<prefix>-NNNNN-of-NNNNN` with matching widths.
    // Rightmost run of digits = "NNNNN" (total shard count).
    let count_start = stem
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    if count_start >= stem.len() {
        return None; // no trailing digits at all
    }
    let count_str = &stem[count_start..];
    let before_count = &stem[..count_start]; // "<prefix>-NNNNN-of-"
    let before_of = before_count.strip_suffix("-of-")?;
    // Then second rightmost digits run = "NNNNN" (this shard's 1-based index).
    let idx_start = before_of
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    if idx_start >= before_of.len() {
        return None;
    }
    let idx_str = &before_of[idx_start..];
    let prefix = before_of[..idx_start].strip_suffix('-')?;

    let this_idx_1based: usize = idx_str.parse().ok()?;
    let total: usize = count_str.parse().ok()?;
    if this_idx_1based == 0 || this_idx_1based > total {
        return None;
    }
    // Width must match across the two numbers (llama.cpp convention).
    if idx_str.len() != count_str.len() {
        return None;
    }
    Some((prefix.to_string(), this_idx_1based - 1, total))
}

/// Discover the full set of sibling shards making up a multi-shard GGUF.
/// `path` is one shard the user pointed at; the returned vec is ordered by
/// shard index (shard 1 first → shard N last) and is guaranteed to be of
/// length `expected_total`.
pub(crate) fn discover_shard_siblings(
    parent: &Path,
    path: &Path,
    expected_total: usize,
) -> Result<Vec<std::path::PathBuf>, ModelError> {
    let (prefix, _, total_from_name) = parse_shard_filename(path).ok_or_else(|| {
        ModelError::Parse(format!(
            "multi-shard GGUF without canonical -NNNNN-of-NNNNN filename: {}",
            path.display()
        ))
    })?;
    if expected_total != total_from_name {
        return Err(ModelError::Parse(format!(
            "shard total mismatch: split.count={expected_total} but filename says of-{total_from_name}",
        )));
    }
    // Detect the widths used in the filename so we reconstruct sibling
    // names byte-for-byte (00001 vs 001).
    let name_str = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let total_width = name_str
        .strip_suffix(".gguf")
        .and_then(|s| s.rsplit("-of-").next())
        .map(|n| n.len())
        .unwrap_or(5);
    let width = name_str
        .strip_suffix(".gguf")
        .and_then(|s| {
            s.strip_suffix(&format!(
                "-of-{expected_total:0>tot_w$}",
                tot_w = total_width
            ))
        })
        .and_then(|s| s.rsplit('-').next())
        .map(|n| n.len())
        .unwrap_or(total_width);

    let mut paths = Vec::with_capacity(expected_total);
    for i in 1..=expected_total {
        let fname = format!(
            "{prefix}-{i:0>idx_width$}-of-{total:0>tot_width$}.gguf",
            prefix = prefix,
            i = i,
            idx_width = width,
            total = expected_total,
            tot_width = total_width,
        );
        let p = parent.join(&fname);
        if !p.exists() {
            return Err(ModelError::Parse(format!(
                "multi-shard GGUF missing expected sibling: {} (looking for shard {} of {})",
                p.display(),
                i,
                expected_total,
            )));
        }
        paths.push(p);
    }
    Ok(paths)
}

impl GgufFile {
    /// Parse a GGUF file header and tensor info (does not read tensor data yet).
    ///
    /// Detects multi-shard splits by checking the `split.count` GGUF metadata
    /// key on the file you point at; when `split.count > 1` (or the filename
    /// matches the canonical `*-NNNNN-of-NNNNN.gguf` pattern), sibling shards
    /// in the same directory are also discovered and their tensor infos are
    /// merged into the returned `GgufFile`. Tensors carry a `shard_idx`
    /// internally so [`Self::load_tensors_filtered`] reads each from the
    /// right shard.
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let mut gguf = Self::open_single(path)?;

        // Multi-shard detection: prefer the explicit `split.*` metadata
        // emitted by llama-gguf-split, fall back to the filename pattern
        // (some splitters skip the metadata).
        let split_count = gguf
            .metadata
            .get("split.count")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let pattern_count = parse_shard_filename(path).map(|(_, _, total)| total);
        let total_shards = match (split_count, pattern_count) {
            (n, _) if n > 1 => n as usize,
            (_, Some(n)) if n > 1 => n,
            _ => return Ok(gguf), // single-file
        };

        // We need every shard in the split — find them all.
        let parent = path.parent().ok_or_else(|| {
            ModelError::Parse(format!("GGUF path has no parent: {}", path.display()))
        })?;
        let shard_paths = discover_shard_siblings(parent, path, total_shards)?;
        debug_assert_eq!(shard_paths.len(), total_shards);

        // The first entry is the shard we already loaded (whichever the
        // caller pointed at). Rewrite `gguf` to be anchored at shard 0 and
        // then accumulate the remaining shards' tensor infos.
        let this_idx = shard_paths.iter().position(|p| p == path).ok_or_else(|| {
            ModelError::Parse(format!(
                "passed shard {} not found in discovered set",
                path.display()
            ))
        })?;
        let mut shards: Vec<ShardInfo> = Vec::with_capacity(total_shards);
        let mut combined_infos: Vec<GgufTensorInfo> = Vec::new();
        for (idx, shard_path) in shard_paths.iter().enumerate() {
            if idx == this_idx {
                shards.push(ShardInfo {
                    path: path.to_path_buf(),
                    data_offset: gguf.data_offset,
                });
                for info in &gguf.tensor_infos {
                    combined_infos.push(GgufTensorInfo {
                        name: info.name.clone(),
                        n_dims: info.n_dims,
                        dims: info.dims.clone(),
                        tensor_type: info.tensor_type,
                        offset: info.offset,
                        shard_idx: idx,
                    });
                }
            } else {
                let other = Self::open_single(shard_path)?;
                shards.push(ShardInfo {
                    path: shard_path.clone(),
                    data_offset: other.data_offset,
                });
                for mut info in other.tensor_infos {
                    info.shard_idx = idx;
                    combined_infos.push(info);
                }
            }
        }

        // Sanity check: total tensor count should match split.tensors.count
        // when that key is emitted (llama-gguf-split always writes it).
        if let Some(expected) = gguf
            .metadata
            .get("split.tensors.count")
            .and_then(|v| v.as_u32())
        {
            if combined_infos.len() != expected as usize {
                return Err(ModelError::Parse(format!(
                    "multi-shard tensor count mismatch: combined {} shards yielded \
                     {} tensors, but split.tensors.count = {}",
                    total_shards,
                    combined_infos.len(),
                    expected
                )));
            }
        }

        gguf.tensor_infos = combined_infos;
        gguf.shards = shards;
        // `gguf.path` / `gguf.data_offset` keep pointing at the
        // user-supplied shard for back-compat with diagnostics; the
        // multi-shard loader uses `shards[info.shard_idx]` internally.
        Ok(gguf)
    }

    /// Open a single GGUF file without multi-shard discovery. Used as the
    /// per-shard primitive by [`Self::open`].
    fn open_single(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut r = Budgeted::new(BufReader::new(file), file_len);

        // Magic
        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(ModelError::Parse(format!(
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})",
                magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(ModelError::Parse(format!(
                "unsupported GGUF version: {version}"
            )));
        }

        let declared_tensors = read_u64(&mut r)?;
        let declared_metadata = read_u64(&mut r)?;
        let n_metadata = checked_count(
            &r,
            declared_metadata,
            MIN_METADATA_ENTRY_BYTES,
            "metadata count",
        )?;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_metadata {
            let key = read_string(&mut r)?;
            let value = read_value(&mut r)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let n_tensors = checked_count(&r, declared_tensors, MIN_TENSOR_INFO_BYTES, "tensor count")?;
        let mut tensor_infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            let dim_count = checked_count(&r, u64::from(n_dims), U64_BYTES, "tensor dims")?;
            let mut dims = Vec::with_capacity(dim_count);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let tensor_type = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensor_infos.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                tensor_type,
                offset,
                shard_idx: 0,
            });
        }

        // Data starts at the next boundary of the declared alignment.
        let alignment = data_alignment(&metadata)?;
        let data_offset = r.position().div_ceil(alignment) * alignment;

        Ok(GgufFile {
            metadata,
            tensor_infos,
            data_offset,
            path: path.to_path_buf(),
            shards: vec![ShardInfo {
                path: path.to_path_buf(),
                data_offset,
            }],
        })
    }
}

#[cfg(test)]
mod tests;
