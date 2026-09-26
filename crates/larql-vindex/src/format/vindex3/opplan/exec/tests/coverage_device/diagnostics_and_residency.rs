//! Diagnostics and residency
//! Geometry checks in front of the device

use super::*;

/// `name` reports the injected name verbatim (it becomes the engine tag
/// on a dump) and `dispatch_stats` counts one submission per device
/// call: a lone `project` is one, a two-matrix f16 FFN is one multi
/// submission plus one for `down`.
#[test]
fn the_name_is_verbatim_and_dispatch_stats_count_one_submission_per_device_call() {
    const NAME: &str = "loop-device-stats";
    const PROJECT_CALLS: u64 = 3;
    /// Up+gate ride one multi submission; down is a second.
    const SUBMISSIONS_PER_GATED_F16_FFN: u64 = 2;

    let backend = DevicePlanBackend::new(LoopDevice, NAME, WeightFormat::F32);
    assert_eq!(backend.name(), NAME);
    let fresh = backend.dispatch_stats().unwrap();
    assert_eq!((fresh.submissions, fresh.device_nanos), (0, 0));

    let w = lcg_values(ROWS * COLS, 1);
    let x = lcg_values(COLS, 2);
    for _ in 0..PROJECT_CALLS {
        project_on(&backend, WeightSlice::F32(&w), &x).unwrap();
    }
    let after_projects = backend.dispatch_stats().unwrap();
    assert_eq!(after_projects.submissions, PROJECT_CALLS);

    let f16 = vec![0u8; FFN_INTERMEDIATE * FFN_HIDDEN * F16_BYTES];
    let x = lcg_values(FFN_HIDDEN, 3);
    backend
        .ffn(ffn_call(
            &x,
            Some(WeightSlice::F16(&f16)),
            WeightSlice::F16(&f16),
            WeightSlice::F16(&f16),
            Activation::Silu,
        ))
        .unwrap();
    let after_ffn = backend.dispatch_stats().unwrap();
    assert_eq!(
        after_ffn.submissions,
        PROJECT_CALLS + SUBMISSIONS_PER_GATED_F16_FFN
    );
}

/// `prepare` hands the device one stream per f16 weight and two per
/// packed 4-bit weight (codes and scales), skipping f32 — the layout
/// the residency hint must see for every byte the decode will touch.
#[test]
fn prepare_wires_one_stream_per_f16_weight_and_two_per_packed_weight() {
    let streams = Arc::new(Mutex::new(Vec::new()));
    let backend = DevicePlanBackend::new(
        WireRecorder {
            streams: Arc::clone(&streams),
        },
        "wire-recorder",
        WeightFormat::F16,
    );
    let f32_weight = lcg_values(ROWS * COLS, 4);
    let f16_bytes = vec![0u8; ROWS * COLS * F16_BYTES];
    let LoadedWeight::Mxfp4 {
        packed: mx_packed,
        scales: mx_scales,
    } = quantize_mxfp4(&f32_weight, ROWS, COLS, "mx").unwrap()
    else {
        unreachable!()
    };
    let LoadedWeight::Nvfp4 {
        packed: nv_packed,
        scales: nv_scales,
        tensor_scale,
        ..
    } = quantize_nvfp4(&f32_weight, ROWS, COLS, "nv").unwrap()
    else {
        unreachable!()
    };
    backend.prepare(&[
        WeightSlice::F32(&f32_weight),
        WeightSlice::F16(&f16_bytes),
        WeightSlice::Mxfp4 {
            packed: mx_packed.as_slice(),
            scales: mx_scales.as_slice(),
        },
        WeightSlice::Nvfp4 {
            packed: nv_packed.as_slice(),
            scales: nv_scales.as_slice(),
            tensor_scale,
            activation: Default::default(),
        },
    ]);
    let seen = streams.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![
            f16_bytes.len(),
            mx_packed.as_slice().len(),
            mx_scales.as_slice().len(),
            nv_packed.as_slice().len(),
            nv_scales.as_slice().len(),
        ],
        "f32 contributes no stream; f16 one; each packed format its codes then its scales"
    );
}

/// An f32 weight whose length is not `out × in` is refused by the seam,
/// naming the geometry, before any kernel sees it.
#[test]
fn a_misshapen_f32_weight_is_refused_naming_the_geometry() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-shape", WeightFormat::F32);
    let short = lcg_values(ROWS * COLS - 1, 5);
    let x = lcg_values(COLS, 6);
    let err = project_on(&backend, WeightSlice::F32(&short), &x).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!("is not [{ROWS}, {COLS}]")),
        "{message}"
    );
}

/// An f16 slice shorter than the matrix is refused by the seam with the
/// byte count — the device (which would also refuse) is never asked, so
/// the message names the seam's check, not a kernel.
#[test]
fn a_short_f16_weight_is_refused_before_reaching_the_device() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-f16-short", WeightFormat::F16);
    let short = vec![0u8; ROWS * COLS * F16_BYTES - F16_BYTES];
    let x = lcg_values(COLS, 7);
    let err = project_on(&backend, WeightSlice::F16(&short), &x).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!(
            "{} weight bytes cannot hold [{ROWS} x {COLS}]",
            short.len()
        )),
        "{message}"
    );
}
