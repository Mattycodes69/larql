//! The storage inventory at a phase end: every backing allocation of
//! continuation state, classified by kind. Split from `measured.rs`
//! (800-line rule).

use larql_vindex::format::vindex3::opplan::exec::continuation::LayerContinuationGeometry;
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

use super::alloc;
use super::inspect::Inspect;
use super::measured::{Backing, F32_BYTES};

pub fn inventory_of<P: Inspect + ?Sized>(
    inner: &mut P,
    geometry: &[LayerContinuationGeometry],
    adopted: &std::collections::HashMap<usize, (usize, u64)>,
) -> Vec<Backing> {
    let mut out = Vec::new();
    for (layer, g) in geometry.iter().enumerate() {
        if let Some(kv) = g.kv_side() {
            let row_payload = kv.kv_dim * F32_BYTES;
            // CODEC-1 C1's hand-off: an inspection read is a read, so it is
            // prepared first (a no-op for providers lending what they hold).
            inner.prepare_layer(layer);
            let view = inner.rows(layer);
            // A row counts as row storage only if the provider adopted it
            // (it is then its own allocation); a row lent from inside a
            // matrix is inventoried once, as the matrix.
            let adopted_rows = |kind: &'static str, row: fn(&KvView<'_>, usize) -> usize| {
                (view.base()..view.end())
                    .filter_map(|p| {
                        let ptr = row(&view, p);
                        adopted.get(&ptr).map(|&(bytes, _)| Backing {
                            kind,
                            layer,
                            index: p,
                            ptr,
                            bytes: Some(bytes),
                            payload_bytes: row_payload,
                        })
                    })
                    .collect::<Vec<_>>()
            };
            let k_rows = adopted_rows("k_row", |v, p| v.key(p).as_ptr() as usize);
            let v_rows = adopted_rows("v_row", |v, p| v.value(p).as_ptr() as usize);
            if !k_rows.is_empty() {
                // The row list itself: a row-backed view's backing address.
                for (kind, ptr) in [
                    ("k_header", view.backing_addresses()[0]),
                    ("v_header", view.backing_addresses()[1]),
                ] {
                    out.push(Backing {
                        kind,
                        layer,
                        index: 0,
                        ptr,
                        bytes: alloc::live_size(ptr),
                        payload_bytes: 0,
                    });
                }
            }
            out.extend(k_rows);
            out.extend(v_rows);
            if let (Some((k, v)), Some(rows)) = (inner.matrix_ptrs(layer), inner.matrix_rows(layer))
            {
                for (kind, ptr) in [("k_matrix", k), ("v_matrix", v)] {
                    out.push(Backing {
                        kind,
                        layer,
                        index: 0,
                        ptr,
                        bytes: alloc::live_size(ptr),
                        payload_bytes: rows * row_payload,
                    });
                }
            }
        }
        if let Some(c) = g.kv_side().and_then(|_| inner.code_list(layer)) {
            out.push(Backing {
                kind: "code_list",
                layer,
                index: 0,
                ptr: c.list,
                bytes: alloc::live_size(c.list),
                payload_bytes: 0,
            });
            for (p, ptr) in (c.base..c.end).zip(inner.code_rows(layer, c.base..c.end)) {
                out.push(Backing {
                    kind: "kv_code",
                    layer,
                    index: p,
                    ptr,
                    bytes: alloc::live_size(ptr),
                    payload_bytes: c.row_bytes,
                });
            }
        }
        if g.recurrent().is_some() {
            let state = inner.recurrent_state(layer).expect("declared recurrent");
            for index in 0..state.len() {
                let cells = state.buffer(index).cells();
                out.push(Backing {
                    kind: "recurrent",
                    layer,
                    index,
                    ptr: cells.as_ptr() as usize,
                    bytes: alloc::live_size(cells.as_ptr() as usize),
                    payload_bytes: cells.len() * F32_BYTES,
                });
            }
        }
        if let Some(latent) = g.latent_kv() {
            let rows = inner.latent_state(layer).expect("declared latent");
            for (index, row) in rows.rows().iter().enumerate() {
                out.push(Backing {
                    kind: "latent_row",
                    layer,
                    index,
                    ptr: row.as_ptr() as usize,
                    bytes: Some(row.capacity() * F32_BYTES),
                    payload_bytes: latent.width * F32_BYTES,
                });
            }
        }
    }
    out
}
