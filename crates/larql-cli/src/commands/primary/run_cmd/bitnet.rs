//! `larql run` on a BitNet b1.58 native-ternary vindex.

use std::io::{self, BufRead, Write};

#[allow(unused_imports)]
use super::*;

/// Serve a BitNet b1.58 native-ternary vindex.
///
/// BitNet uses a separate ternary forward (`larql_inference::ternary`) rather
/// than the dense engine dispatch, so it bypasses the `walk_cmd` path that the
/// dense `run_once` / `run_chat` delegate to. Greedy streaming generation;
/// no prompt drops into a simple stdin REPL. EOS is sourced the same way as
/// the dense path (`EosConfig::from_vindex_dir`).
///
/// MVP scope (P-A): raw prompt encode (no chat-template rendering yet) and
/// greedy sampling. Chat-template + sampling-flag parity with the dense path
/// are follow-ups; see ROADMAP "Productization plan" P-A.
pub(super) fn run_bitnet(
    vindex_path: &std::path::Path,
    args: &RunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::generate::{eos::EosConfig, SamplingConfig};
    use larql_inference::ternary;

    let model = ternary::load_bitnet_model(vindex_path)?;
    let tokenizer =
        larql_vindex::tokenizers::Tokenizer::from_file(vindex_path.join("tokenizer.json"))
            .map_err(|e| format!("load tokenizer.json: {e}"))?;
    let eos = EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.max_tokens;
    let verbose = args.verbose;

    let generate_one = |prompt: &str| -> Result<(), Box<dyn std::error::Error>> {
        let enc = tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("encode prompt: {e}"))?;
        let mut stdout = io::stdout();
        let emitted = ternary::generate_streaming_bitnet(
            &model,
            &tokenizer,
            enc.get_ids(),
            max_tokens,
            SamplingConfig::greedy(),
            &eos,
            |_id, delta, _ms| {
                print!("{delta}");
                let _ = stdout.flush();
            },
        );
        println!();
        if verbose {
            eprintln!("[bitnet] {emitted} tokens");
        }
        Ok(())
    };

    if let Some(prompt) = args.prompt.as_deref() {
        return generate_one(prompt);
    }

    // Chat REPL (single-turn, no multi-turn template — BitNet MVP).
    eprintln!(
        "larql chat (bitnet) — {} (Ctrl-D to exit)",
        vindex_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
    );
    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush()?;
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                eprintln!();
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(Box::new(e)),
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if let Err(e) = generate_one(prompt) {
            eprintln!("Error: {e}");
        }
    }
}
