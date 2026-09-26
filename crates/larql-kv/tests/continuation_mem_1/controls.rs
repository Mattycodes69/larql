//! controls

use super::*;

/// Negative: with no scope open, allocations are never attributed.
/// Recorder silence: an empty scope attributes exactly nothing.
#[test]
fn control_negative_and_recorder_silence() {
    let _serial = serial();
    let before = alloc::attributed_totals();
    let planted: Vec<Vec<u8>> = (0..64).map(|i| vec![0u8; 16 + i]).collect();
    drop(planted);
    assert_eq!(
        alloc::attributed_totals(),
        before,
        "no scope open, nothing attributed"
    );

    let empty = alloc::enter().leave();
    assert_eq!(empty.allocs, 0, "an empty scope attributes no allocation");
    assert_eq!(empty.frees, 0);
    assert_eq!(empty.events, 0);
}

/// Positive: a scope does see an allocation made inside it, with its size.
#[test]
fn control_positive_attribution() {
    let _serial = serial();
    let scope = alloc::enter();
    let v: Vec<f32> = Vec::with_capacity(33);
    let delta = scope.leave();
    assert_eq!(delta.allocs, 1);
    assert_eq!(delta.alloc_bytes, 33 * 4);
    assert_eq!(alloc::live_size(v.as_ptr() as usize), Some(33 * 4));
    let ptr = v.as_ptr() as usize;
    drop(v);
    assert_eq!(
        alloc::live_size(ptr),
        None,
        "a free forgets the allocation wherever it happens"
    );
}

/// Thread integrity: an allocation on another thread while a scope is
/// open is counted foreign (the run would be INVALID).
#[test]
fn control_thread_integrity() {
    let _serial = serial();
    let handle = std::thread::spawn(|| {
        std::thread::park();
        let planted = vec![7u8; 4096];
        std::hint::black_box(&planted);
    });
    let scope = alloc::enter();
    handle.thread().unpark();
    // `join` itself does not allocate; the planted Vec is on the worker.
    let joined = handle.join();
    let delta = scope.leave();
    joined.unwrap();
    assert!(
        delta.foreign >= 1,
        "the planted foreign allocation must be seen: {delta:?}"
    );

    // The owner's own allocation is attributed to the owner — counted,
    // sized and logged. (Whether some other live thread also allocates
    // during the scope is not this control's claim: it is exactly what the
    // foreign counter exists to report, and a process has other threads.)
    let clean = alloc::enter();
    let local = vec![1u8; 64];
    let delta = clean.leave();
    let events = alloc::events(&delta);
    drop(local);
    assert_eq!(
        delta.allocs, 1,
        "the owner's allocation is counted once: {delta:?}"
    );
    assert_eq!(delta.alloc_bytes, 64);
    assert!(
        events
            .iter()
            .any(|e| e.kind == alloc::EventKind::Alloc && e.new_size == 64),
        "the owner's allocation is in the owner's event log"
    );
}

/// Tags survive address reuse: an allocation born in a tagged scope, freed,
/// and its address reused by an untagged scope's allocation, is no longer
/// reported as the tagged one (VIEW-1 V3: the append-born anti-cheat).
#[test]
fn control_tag_survives_address_reuse() {
    let _serial = serial();
    let tagged = alloc::enter_tagged(measured::APPEND_TAG);
    let first = vec![1u8; 96];
    let _ = tagged.leave();
    let address = first.as_ptr() as usize;
    assert_eq!(
        alloc::live_size_tagged(address, measured::APPEND_TAG),
        Some(96)
    );
    drop(first);
    assert_eq!(alloc::live_size_tagged(address, measured::APPEND_TAG), None);
    let untagged = alloc::enter();
    let second = vec![2u8; 96];
    let _ = untagged.leave();
    if second.as_ptr() as usize == address {
        assert_eq!(alloc::live_size(address), Some(96), "the reuse is live");
        assert_eq!(
            alloc::live_size_tagged(address, measured::APPEND_TAG),
            None,
            "a reused address must not answer as the append-born allocation"
        );
    }
    drop(second);
}

/// Realloc classification: every growth is classified moved or in place
/// by pointer comparison, and moved bytes are the old sizes of the moves.
#[test]
fn control_realloc_classification() {
    let _serial = serial();
    let mut v: Vec<u8> = Vec::with_capacity(1);
    v.push(0);
    let scope = alloc::enter();
    for cap in (1..=22).map(|s| 1usize << s) {
        v.reserve_exact(cap - v.len());
        v.resize(cap, 0);
    }
    let delta = scope.leave();
    let events = alloc::events(&delta);
    let mut moved_bytes = 0u64;
    let (mut moved, mut in_place) = (0, 0);
    for e in &events {
        match e.kind {
            alloc::EventKind::ReallocMoved => {
                assert_ne!(e.old_ptr, e.new_ptr);
                moved += 1;
                moved_bytes += e.old_size as u64;
            }
            alloc::EventKind::ReallocInPlace => {
                assert_eq!(e.old_ptr, e.new_ptr);
                in_place += 1;
            }
            _ => {}
        }
    }
    assert_eq!(moved + in_place, 22, "every growth is a realloc");
    assert_eq!(delta.moved_bytes, moved_bytes);
    assert!(moved > 0, "a small-to-large growth must move at least once");
    eprintln!("realloc control: {moved} moved, {in_place} in place");
}
