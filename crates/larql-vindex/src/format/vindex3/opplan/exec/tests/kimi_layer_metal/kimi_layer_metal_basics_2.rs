use super::*;

/// **Q8_0 sits strictly ABOVE Q6_K on the fidelity axis at a real
/// layer.** Both quantisations of all three projections move the layer
/// output — they are real representations, not pass-throughs — and
/// Q8_0's displacement is strictly smaller than Q6_K's on the same
/// weights and input.
///
/// This is the smoke gate the precision ladder stands on: if an ~8-bit
/// bank did NOT beat Q6_K here, running teacher-forced quality banks
/// against it would measure a defect, not a representation.
#[test]
fn q8_0_displaces_the_layer_output_less_than_q6_k() {
    let Some((b, f)) = setup() else {
        return;
    };
    let per = f.inter * f.hidden * 2;
    let per_down = f.hidden * f.inter * 2;

    let state_ref = KdaDeviceState::zeros(&b, f.shape());
    let (want, _) = b
        .kimi_decoder_layer(f.layer(&state_ref), &f.x)
        .expect("reference layer runs");

    // One closure per encoding, differing ONLY in the re-encoder, so
    // the two arms cannot drift in table construction or bank layout.
    type Requant<'a> = &'a dyn Fn(&[u8], usize, usize, usize) -> (Vec<u8>, usize);
    let run = |requant_gate_up: Requant, requant_down: Requant, enc: ExpertEncoding| -> Vec<f32> {
        let (g, q_per) = requant_gate_up(&f.bank_gate, per, f.inter, f.hidden);
        let (u, _) = requant_gate_up(&f.bank_up, per, f.inter, f.hidden);
        let (d, q_per_down) = requant_down(&f.bank_down, per_down, f.hidden, f.inter);
        let (sg, _) = requant_gate_up(&f.shared_gate, per, f.inter, f.hidden);
        let (su, _) = requant_gate_up(&f.shared_up, per, f.inter, f.hidden);
        let (sd, _) = requant_down(&f.shared_down, per_down, f.hidden, f.inter);
        let identity: Vec<usize> = (0..f.bank_gate.len() / per).collect();
        let t = table(&f.residency, per, q_per, &identity);
        let td = table(&f.residency, per_down, q_per_down, &identity);
        let state = KdaDeviceState::zeros(&b, f.shape());
        let mut w = f.layer(&state);
        if let FfnSpec::Moe(m) = &mut w.ffn {
            fn bank<'a>(
                bytes: &'a [u8],
                tbl: &'a [u32],
                shared: &'a [u8],
                enc: ExpertEncoding,
            ) -> ProjectionBank<'a> {
                ProjectionBank {
                    routed: EncodedRegion {
                        bytes,
                        encoding: enc,
                    },
                    addressing: ExpertAddressing::Table(tbl),
                    shared: Some(EncodedRegion {
                        bytes: shared,
                        encoding: enc,
                    }),
                }
            }
            m.gate = bank(&g, &t, &sg, enc);
            m.up = bank(&u, &t, &su, enc);
            m.down = bank(&d, &td, &sd, enc);
            let (out, _) = b.kimi_decoder_layer(w, &f.x).expect("quantised layer runs");
            return out;
        }
        unreachable!("the fixture layer is routed");
    };

    let q8 = run(&to_q8_0_blocks, &to_q8_0_blocks, ExpertEncoding::Q80);
    let q6 = run(&to_q6k_blocks, &to_q6k_blocks, ExpertEncoding::Q6K);

    let rel = |a: &[f32], c: &[f32]| {
        let se: f64 = a.iter().zip(c).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let ss: f64 = c.iter().map(|y| (*y as f64).powi(2)).sum();
        (se / ss).sqrt()
    };
    let (e8, e6) = (rel(&q8, &want), rel(&q6, &want));
    eprintln!("[q8] all-projection displacement vs BF16: Q8_0 {e8:.3e}, Q6_K {e6:.3e}");
    assert!(
        e8 > 0.0,
        "Q8_0 must move the answer, or this proves nothing about its decode"
    );
    assert!(
        e8 < e6,
        "Q8_0 ({e8:.3e}) must displace the output LESS than Q6_K ({e6:.3e}) on the same \
         weights, or the ladder's middle rung is below its bottom one"
    );
}
