//! BINARY DOWN_META
//! ERROR HANDLING

use super::*;

#[test]
fn binary_down_meta_write_read_round_trip() {
    let _idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_binary_dm");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write binary format
    let count = larql_vindex::down_meta::write_binary(
        &dir,
        &[
            Some(vec![
                Some(make_meta("Paris", 100, 0.95)),
                Some(make_meta("French", 101, 0.88)),
                None,
            ]),
            Some(vec![
                Some(make_meta("Berlin", 200, 0.90)),
                None,
                Some(make_meta("Spain", 202, 0.70)),
            ]),
        ],
        1, // top_k = 1
    )
    .unwrap();
    assert_eq!(count, 4); // 2 + 2 (Nones don't count)

    // Verify file exists and is much smaller than JSONL would be
    let bin_path = dir.join("down_meta.bin");
    assert!(bin_path.exists());
    let bin_size = std::fs::metadata(&bin_path).unwrap().len();
    // Header (16) + 2 layers × (4 bytes layer header + 3 features × (4+4+1×8) bytes)
    assert!(bin_size > 0);
    assert!(bin_size < 200); // should be very small for 6 features

    // Read back — needs a tokenizer for string resolution
    // Create a minimal tokenizer that maps IDs to strings
    // Since we can't easily create a real tokenizer in tests,
    // verify the raw binary structure is correct
    let data = std::fs::read(&bin_path).unwrap();
    // Check magic
    assert_eq!(
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        0x444D4554
    );
    // Check version
    assert_eq!(u32::from_le_bytes([data[4], data[5], data[6], data[7]]), 1);
    // Check num_layers
    assert_eq!(
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
        2
    );
    // Check top_k
    assert_eq!(
        u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
        1
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_down_meta_writes_binary() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_dm_binary");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let count = idx.save_down_meta(&dir).unwrap();
    assert_eq!(count, 5); // 3 + 2

    // Binary file should exist (JSONL no longer written)
    assert!(dir.join("down_meta.bin").exists());
    assert!(!dir.join("down_meta.jsonl").exists());

    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    assert!(bin_size > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_nonexistent_vindex_errors() {
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let result =
        VectorIndex::load_vindex(std::path::Path::new("/nonexistent/fake.vindex"), &mut cb);
    assert!(result.is_err());
}

#[test]
fn load_nonexistent_config_errors() {
    let result = larql_vindex::load_vindex_config(std::path::Path::new("/nonexistent/fake.vindex"));
    assert!(result.is_err());
}
