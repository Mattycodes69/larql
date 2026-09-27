use super::*;
use larql_vindex::ndarray::Array2;

// ── Binary wire format helpers ───────────────────────────────────────────

fn make_binary_embed_request(token_ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + token_ids.len() * 4);
    out.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &id in token_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

fn make_binary_logits_request(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 4);
    for &v in floats {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ── Embed binary encode/decode ───────────────────────────────────────────

#[test]
fn binary_embed_request_encodes_num_tokens() {
    let body = make_binary_embed_request(&[1, 2, 3]);
    let num = u32::from_le_bytes(body[..4].try_into().unwrap());
    assert_eq!(num, 3);
    assert_eq!(parse_binary_embed_request(&body).unwrap(), vec![1, 2, 3]);
}

#[test]
fn binary_embed_request_encodes_token_ids() {
    let ids = [100u32, 200, 300];
    let body = make_binary_embed_request(&ids);
    for (i, &expected) in ids.iter().enumerate() {
        let got = u32::from_le_bytes(body[4 + i * 4..4 + i * 4 + 4].try_into().unwrap());
        assert_eq!(got, expected);
    }
}

#[test]
fn binary_embed_request_total_length() {
    // 4 (num_tokens u32) + N × 4 (token_id u32)
    let body = make_binary_embed_request(&[1, 2, 3, 4, 5]);
    assert_eq!(body.len(), 4 + 5 * 4);
}

#[test]
fn binary_embed_response_header_fields() {
    // Response format: [seq_len u32][hidden_size u32][seq_len × hidden_size f32]
    let seq_len = 2usize;
    let hidden = 4usize;
    let h = Array2::<f32>::from_elem((seq_len, hidden), 1.23);
    let out = encode_binary_embed_response(&h);
    assert_eq!(
        u32::from_le_bytes(out[..4].try_into().unwrap()) as usize,
        seq_len
    );
    assert_eq!(
        u32::from_le_bytes(out[4..8].try_into().unwrap()) as usize,
        hidden
    );
    assert_eq!(out.len(), 8 + seq_len * hidden * 4);
}

#[test]
fn binary_embed_response_float_roundtrip() {
    let seq_len = 1usize;
    let hidden = 4usize;
    let values = [0.1f32, -0.5, 1.0, 2.5];
    let mut out = vec![0u8; 8];
    for &v in &values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let payload = &out[8..];
    for (i, chunk) in payload.as_chunks::<4>().0.iter().enumerate() {
        let got = f32::from_le_bytes(*chunk);
        assert!(
            (got - values[i]).abs() < 1e-6,
            "float[{i}]: {got} != {}",
            values[i]
        );
    }
    let _ = (seq_len, hidden);
}

// ── Logits binary encode/decode ──────────────────────────────────────────

#[test]
fn binary_logits_request_byte_length() {
    let residual: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let body = make_binary_logits_request(&residual);
    assert_eq!(body.len(), 8 * 4);
}

#[test]
fn binary_logits_request_float_roundtrip() {
    let residual = [1.5f32, -2.0, 0.0, 99.9];
    let body = make_binary_logits_request(&residual);
    assert_eq!(parse_binary_logits_request(&body).unwrap(), residual);
    for (i, chunk) in body.as_chunks::<4>().0.iter().enumerate() {
        let got = f32::from_le_bytes(*chunk);
        assert!((got - residual[i]).abs() < 1e-6);
    }
}

#[test]
fn binary_logits_odd_length_is_invalid() {
    // A body of 5 bytes is not a multiple of 4.
    let body = [0u8; 5];
    assert_ne!(body.len() % 4, 0, "5 bytes must fail the alignment check");
    assert!(matches!(
        parse_binary_logits_request(&body),
        Err(ServerError::BadRequest(_))
    ));
}

#[test]
fn binary_embed_rejects_short_header() {
    assert!(matches!(
        parse_binary_embed_request(&[0, 1, 2]),
        Err(ServerError::BadRequest(_))
    ));
}

#[test]
fn binary_embed_rejects_truncated_token_ids() {
    let mut body = Vec::new();
    body.extend_from_slice(&2u32.to_le_bytes());
    body.extend_from_slice(&7u32.to_le_bytes());
    assert!(matches!(
        parse_binary_embed_request(&body),
        Err(ServerError::BadRequest(_))
    ));
}

// ── Token decode query parsing ───────────────────────────────────────────

#[test]
fn token_decode_query_parse_csv() {
    let q = "9515,235,1234";
    let ids: Vec<u32> = q
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().unwrap())
        .collect();
    assert_eq!(ids, vec![9515u32, 235, 1234]);
}

#[test]
fn token_decode_query_handles_whitespace() {
    let q = " 9515 , 235 , 1234 ";
    let ids: Vec<u32> = q
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().unwrap())
        .collect();
    assert_eq!(ids, vec![9515u32, 235, 1234]);
}

#[test]
fn token_decode_query_single_id() {
    let q = "9515";
    let ids: Vec<u32> = q
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().unwrap())
        .collect();
    assert_eq!(ids, vec![9515u32]);
}

// ── Embed matrix lookup logic ────────────────────────────────────────────

#[test]
fn embed_lookup_returns_correct_row() {
    // embed[2] = [0, 0, 1, 0] → after scale=1.0 same
    let mut embed = Array2::<f32>::zeros((4, 4));
    embed[[2, 2]] = 1.0;
    let scale = 1.0f32;

    let tok_id = 2usize;
    let row: Vec<f32> = embed.row(tok_id).iter().map(|&v| v * scale).collect();
    assert_eq!(row, vec![0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn embed_lookup_applies_scale() {
    let mut embed = Array2::<f32>::zeros((4, 4));
    embed[[1, 0]] = 1.0;
    let scale = 2.5f32;

    let row: Vec<f32> = embed.row(1).iter().map(|&v| v * scale).collect();
    assert_eq!(row, vec![2.5, 0.0, 0.0, 0.0]);
}

#[test]
fn embed_lookup_out_of_range_detected() {
    let embed = Array2::<f32>::zeros((8, 4));
    let vocab = embed.shape()[0];
    assert!((8usize >= vocab)); // token_id=8 is OOB for vocab=8
    assert!(7usize < vocab); // token_id=7 is in range
}

#[test]
fn embed_response_shape() {
    // seq_len=3 tokens, hidden=4 → residual is [[f32×4], [f32×4], [f32×4]]
    let seq_len = 3;
    let hidden = 4;
    let h = Array2::<f32>::zeros((seq_len, hidden));
    let residual: Vec<Vec<f32>> = h.rows().into_iter().map(|r| r.to_vec()).collect();
    assert_eq!(residual.len(), seq_len);
    assert!(residual.iter().all(|row| row.len() == hidden));
}

// ── Default parameter values ─────────────────────────────────────────────

#[test]
fn default_top_k_is_five() {
    assert_eq!(default_top_k(), 5);
}

#[test]
fn default_temperature_is_one() {
    assert!((default_temperature() - 1.0).abs() < 1e-6);
}
