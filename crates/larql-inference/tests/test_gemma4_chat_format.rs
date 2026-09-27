//! Gemma 4's fallback chat format against Google's own templates.
//!
//! Every official Gemma 4 `chat_template.jinja` under
//! `LARQL_GEMMA4_TEMPLATE_DIR` (searched recursively — point it at a Hugging
//! Face hub cache) is rendered through larql's Jinja renderer and compared
//! with the fallback. The turn syntax must match exactly. The only allowed
//! difference is the empty thought channel some checkpoints open the model
//! turn with, which is a checkpoint fact the fallback deliberately omits
//! (`ChatFormat::Gemma4Turns`). `#[ignore]`d: the templates are not in-tree.
//!
//! ```sh
//! LARQL_GEMMA4_TEMPLATE_DIR=~/.cache/huggingface/hub \
//!   cargo test -p larql-inference --test test_gemma4_chat_format -- --ignored
//! ```

use std::path::{Path, PathBuf};

use larql_inference::chat::render_chat_template_multi;
use larql_inference::prompt::ChatTemplate;
use larql_models::detect::ChatFormat;

const TEMPLATE_DIR_ENV: &str = "LARQL_GEMMA4_TEMPLATE_DIR";
const BOS: &str = "<bos>";
const THOUGHT_TAIL: &str = "<|channel>thought\n<channel|>";

/// Official Gemma 4 templates: `chat_template.jinja` files using the
/// Gemma 4 turn markers.
fn official_templates(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == "chat_template.jinja")
                && std::fs::read_to_string(&path).is_ok_and(|t| t.contains("<|turn>"))
            {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn render(template: &str, messages: &[(&str, &str)]) -> String {
    let cfg = serde_json::json!({ "bos_token": BOS });
    let messages: Vec<(String, String)> = messages
        .iter()
        .map(|(r, c)| (r.to_string(), c.to_string()))
        .collect();
    render_chat_template_multi(template, &cfg, &messages, false).expect("template renders")
}

/// The official rendering with the checkpoint-specific thought channel
/// removed from the end, if present.
fn without_thought_tail(rendered: &str) -> &str {
    rendered.strip_suffix(THOUGHT_TAIL).unwrap_or(rendered)
}

#[test]
#[ignore = "needs official Gemma 4 templates; set LARQL_GEMMA4_TEMPLATE_DIR"]
fn fallback_matches_every_official_gemma4_template() {
    let dir =
        PathBuf::from(std::env::var(TEMPLATE_DIR_ENV).expect("set LARQL_GEMMA4_TEMPLATE_DIR"));
    let templates = official_templates(&dir);
    assert!(
        !templates.is_empty(),
        "no Gemma 4 chat_template.jinja under {}",
        dir.display()
    );
    let fallback = ChatFormat::Gemma4Turns
        .fallback_template()
        .expect("Gemma 4 has a fallback template");
    let conversations: [&[(&str, &str)]; 2] = [
        &[("user", "Hi")],
        &[("system", "Be brief."), ("user", "Hi")],
    ];
    for path in &templates {
        let official = std::fs::read_to_string(path).unwrap();
        for messages in conversations {
            let theirs = render(&official, messages);
            let ours = render(fallback, messages);
            assert_eq!(
                without_thought_tail(&theirs),
                ours,
                "{} {messages:?}",
                path.display()
            );
            // The single-prompt wrap and the turn renderer agree with the
            // official template too (BOS is the tokenizer's to add).
            let rendered = ChatTemplate::Gemma4.render_messages(messages.iter().copied());
            assert_eq!(
                without_thought_tail(&theirs).strip_prefix(BOS),
                Some(rendered.as_str()),
                "{} {messages:?}",
                path.display()
            );
        }
        let wrapped = ChatTemplate::Gemma4.wrap("Hi");
        let theirs = render(&official, &[("user", "Hi")]);
        assert_eq!(
            without_thought_tail(&theirs).strip_prefix(BOS),
            Some(wrapped.as_str()),
            "{}",
            path.display()
        );
    }
    eprintln!("checked {} official Gemma 4 templates", templates.len());
}
