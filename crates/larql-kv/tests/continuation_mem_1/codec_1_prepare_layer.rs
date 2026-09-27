//! CONTINUATION-CODEC-1 C1: the harness forwards and records `prepare_layer`

use super::*;

/// Every `rows(layer)` the executor makes is immediately preceded by
/// `prepare_layer(layer)`, and `Measured` forwards and records the hook —
/// so C3's instrument sees the call a compressed provider depends on.
#[test]
fn every_read_is_preceded_by_its_layers_prepare_in_the_record() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9],
        decode: vec![1, 2, 3, 4],
    };
    let mut row = Measured::new(RowKvState::default());
    subjects::run(&subject, &ops, &backend, &mut row, &journey);
    let (calls, _) = row.take_records();
    let reads: Vec<usize> = (0..calls.len())
        .filter(|&i| calls[i].method == measured::Method::Rows)
        .collect();
    assert!(!reads.is_empty(), "resume and decode read rows");
    for i in reads {
        let before = i.checked_sub(1).map(|j| &calls[j]);
        assert!(
            before.is_some_and(
                |c| c.method == measured::Method::PrepareLayer && c.layer == calls[i].layer
            ),
            "read {i} of layer {:?} was not preceded by its prepare_layer",
            calls[i].layer
        );
    }
}
