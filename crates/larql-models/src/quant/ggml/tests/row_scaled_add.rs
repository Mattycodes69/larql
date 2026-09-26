use super::*;

#[test]
fn q4k_row_scaled_add_matches_alpha_times_deq() {
    let data = synth_q4k_block(13);
    let alpha = 0.25_f32;
    let deq = dequantize_q4_k(&data, 256).unwrap();
    let mut out = vec![0.0f32; 256];
    q4k_row_scaled_add(&data, alpha, &mut out).unwrap();
    for (i, (&o, &d)) in out.iter().zip(&deq).enumerate() {
        let expected = alpha * d;
        assert!(
            (o - expected).abs() < 1e-5,
            "idx {i}: got {o} expected {expected}"
        );
    }
}

#[test]
fn q6k_row_scaled_add_matches_alpha_times_deq() {
    let data = synth_q6k_block(21);
    let alpha = 0.5_f32;
    let deq = dequantize_q6_k(&data, 256).unwrap();
    let mut out = vec![0.0f32; 256];
    q6k_row_scaled_add(&data, alpha, &mut out).unwrap();
    for (i, (&o, &d)) in out.iter().zip(&deq).enumerate() {
        let expected = alpha * d;
        assert!(
            (o - expected).abs() < 1e-5,
            "idx {i}: got {o} expected {expected}"
        );
    }
}

#[test]
fn q4k_row_scaled_add_rejects_misaligned() {
    let mut out = vec![0.0f32; 300]; // not a multiple of 256
    match q4k_row_scaled_add(&[0u8; 144], 1.0, &mut out) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q6k_row_scaled_add_rejects_misaligned() {
    let mut out = vec![0.0f32; 300];
    match q6k_row_scaled_add(&[0u8; 210], 1.0, &mut out) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q4k_row_dot_rejects_misaligned_x_length() {
    let x = vec![0.0f32; 200];
    match q4k_row_dot(&[0u8; 144], &x) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q4k_row_dot_rejects_short_data() {
    let x = vec![0.0f32; 256];
    // n_blocks = 1 requires 144 bytes; supply 16.
    match q4k_row_dot(&[0u8; 16], &x) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q4k_row_scaled_add_rejects_short_data() {
    let mut out = vec![0.0f32; 256];
    match q4k_row_scaled_add(&[0u8; 16], 1.0, &mut out) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q6k_row_dot_rejects_misaligned_x_length() {
    // x length 200 is not a multiple of 256.
    let x = vec![0.0f32; 200];
    match q6k_row_dot(&[0u8; 210], &x) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q6k_row_dot_rejects_short_data() {
    // n_blocks = 1 requires 210 bytes; supply 16.
    let x = vec![0.0f32; 256];
    match q6k_row_dot(&[0u8; 16], &x) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q6k_row_scaled_add_rejects_short_data() {
    let mut out = vec![0.0f32; 256];
    match q6k_row_scaled_add(&[0u8; 16], 1.0, &mut out) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

/// Cover `q4k_row_scaled_add_scalar` directly — on aarch64 the
/// production path goes through the NEON variant so the scalar
/// reference is `#[allow(dead_code)]`.
#[test]
fn q4k_row_scaled_add_scalar_matches_alpha_times_deq() {
    use super::super::q4_k::q4k_row_scaled_add_scalar;
    let data = synth_q4k_block(17);
    let alpha = 0.375_f32;
    let deq = dequantize_q4_k(&data, 256).unwrap();
    let mut out = vec![0.0f32; 256];
    q4k_row_scaled_add_scalar(&data, alpha, &mut out, 1);
    for (i, (&o, &d)) in out.iter().zip(&deq).enumerate() {
        let expected = alpha * d;
        assert!(
            (o - expected).abs() < 1e-5,
            "idx {i}: got {o} expected {expected}"
        );
    }
}
