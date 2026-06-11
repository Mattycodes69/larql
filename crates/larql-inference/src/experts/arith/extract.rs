//! Extraction for the arithmetic expert (spec §4).
//!
//! **Explicit path:** symbolic parse of operand digit spans and operators off
//! the prompt surface — exact by construction, zero tokens. The same scanner
//! is the tier-0 gate (`gate.rs`), so a tier-0 fire implies the symbolic
//! extract succeeds: fire ⇒ extraction, the A10 invariant, holds by
//! construction.
//!
//! **Disguised path:** 2-shot rewrite prompt → parse the *emitted expression*.
//! The parser reads the model's expression, never its sum (rigging-proofed by
//! design — anything after `=` is discarded).

use super::alu::{BigInt, Expr, Op};

/// One lexed token of the prompt surface. `Other` breaks operand/operator
/// adjacency so unrelated numbers never join into an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Num(String),
    Op(Op),
    Other,
}

/// Scan `text` for the longest explicit integer chain `N op N (op N)*`.
///
/// Operator rules (distractor protection — gate specificity is the contract):
/// - `+`, `*`, `×`, `−` (U+2212) count anywhere between digit spans;
/// - ASCII `-` counts only with whitespace on both sides, so dates
///   (`2026-06-11`), ranges (`5-10`) and phone formats never fire;
/// - `x`/`X` counts only as a standalone word between digit spans (`3 x 4`);
/// - `/` never counts — division is OPEN in v0.1 and `06/11` is a date.
///
/// Numbers absorb `1,234,567`-style thousands separators and `_` separators.
pub fn find_expression(text: &str) -> Option<Expr> {
    let toks = lex(text);
    let mut best: Option<(usize, usize)> = None; // (start, op_count)

    let mut i = 0;
    while i < toks.len() {
        if matches!(toks[i], Tok::Num(_)) {
            let mut j = i;
            let mut ops = 0usize;
            while matches!(toks.get(j + 1), Some(Tok::Op(_)))
                && matches!(toks.get(j + 2), Some(Tok::Num(_)))
            {
                ops += 1;
                j += 2;
            }
            if ops > 0 && best.map(|(_, b)| ops > b).unwrap_or(true) {
                best = Some((i, ops));
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }

    let (start, op_count) = best?;
    let mut operands = Vec::with_capacity(op_count + 1);
    let mut ops = Vec::with_capacity(op_count);
    for k in 0..=op_count {
        let Tok::Num(s) = &toks[start + 2 * k] else {
            return None;
        };
        operands.push(BigInt::parse(s)?);
        if k < op_count {
            let Tok::Op(op) = &toks[start + 2 * k + 1] else {
                return None;
            };
            ops.push(*op);
        }
    }
    Some(Expr { operands, ops })
}

fn lex(text: &str) -> Vec<Tok> {
    let chars: Vec<char> = text.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            let mut num = String::new();
            while i < chars.len() {
                let c = chars[i];
                if c.is_ascii_digit() {
                    num.push(c);
                    i += 1;
                } else if (c == ',' || c == '_')
                    && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())
                {
                    // Separator inside a number; keep digits only.
                    i += 1;
                } else {
                    break;
                }
            }
            toks.push(Tok::Num(num));
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let ws_before = i == 0 || chars[i - 1].is_whitespace();
        let ws_after = i + 1 >= chars.len() || chars[i + 1].is_whitespace();
        let op = match c {
            '+' => Some(Op::Add),
            '*' | '×' => Some(Op::Mul),
            '−' => Some(Op::Sub),
            '-' if ws_before && ws_after => Some(Op::Sub),
            'x' | 'X' if ws_before && ws_after => Some(Op::Mul),
            _ => None,
        };
        match op {
            Some(op) => toks.push(Tok::Op(op)),
            None => toks.push(Tok::Other),
        }
        i += 1;
    }
    toks
}

/// The 2-shot rewrite prompt (the measured A8 floor — deliberately untuned;
/// structured-output extraction is the OPEN improvement, not this prompt).
pub fn rewrite_prompt(question: &str) -> String {
    format!(
        "Rewrite each question as a bare arithmetic expression. Do not solve it.\n\
         Q: If you have 7 apples and pick 5 more, how many apples do you have?\n\
         E: 7 + 5\n\
         Q: A crate holds 240 bottles. How many bottles are in 3 crates?\n\
         E: 240 * 3\n\
         Q: {question}\n\
         E:"
    )
}

/// Parse the model-emitted rewrite. First emitted line only, truncated at
/// `=` so the model's own sum — if it volunteers one — is never consumed.
pub fn parse_rewrite(generated: &str) -> Option<Expr> {
    let line = generated.trim_start().lines().next()?;
    let line = line.split(['=', '\u{ff1d}']).next().unwrap_or(line);
    find_expression(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Option<String> {
        find_expression(text).map(|e| format!("{e} -> {}", e.eval()))
    }

    #[test]
    fn explicit_forms_parse_exactly() {
        assert_eq!(parse("12 + 7 ="), Some("12 + 7 -> 19".into()));
        assert_eq!(
            parse("What is 123456 + 654321?"),
            Some("123456 + 654321 -> 777777".into())
        );
        assert_eq!(parse("12345 * 6789"), Some("12345 * 6789 -> 83810205".into()));
        assert_eq!(parse("12×34"), Some("12 * 34 -> 408".into()));
        assert_eq!(parse("3 x 4"), Some("3 * 4 -> 12".into()));
        assert_eq!(parse("100000 - 1 ="), Some("100000 - 1 -> 99999".into()));
        assert_eq!(parse("47−5"), Some("47 - 5 -> 42".into()));
    }

    #[test]
    fn two_op_chains_parse() {
        assert_eq!(
            parse("999 + 111 - 222 ="),
            Some("999 + 111 - 222 -> 888".into())
        );
        assert_eq!(parse("2 + 3 * 4"), Some("2 + 3 * 4 -> 14".into()));
    }

    #[test]
    fn thousands_separators_absorb_into_one_operand() {
        assert_eq!(
            parse("1,234,567 + 1"),
            Some("1234567 + 1 -> 1234568".into())
        );
        assert_eq!(parse("1_000 + 24"), Some("1000 + 24 -> 1024".into()));
    }

    #[test]
    fn expression_stops_at_equals_never_reads_the_sum() {
        // The chain is "12 + 7"; the 19 after '=' must not join it.
        let e = find_expression("12 + 7 = 19").expect("expr");
        assert_eq!(e.to_string(), "12 + 7");
    }

    #[test]
    fn distractors_do_not_parse() {
        for text in [
            "My phone number is 4415550172.",
            "The meeting is on 2026-06-11.",
            "Trains depart at 18:45 from platform 3.",
            "Order 66 was executed in 19 BBY.",
            "What is the capital of France?",
            "Account 123456789012345678901234567890 is active.",
            "It takes 5-10 days to ship.",
            "The score was 3/4.",
            "version 1.2.3 released",
        ] {
            assert!(
                find_expression(text).is_none(),
                "false parse on distractor: {text:?}"
            );
        }
    }

    #[test]
    fn hyphen_needs_whitespace_both_sides() {
        assert!(find_expression("100-1").is_none());
        assert!(find_expression("100- 1").is_none());
        assert!(find_expression("100 -1").is_none());
        assert!(find_expression("100 - 1").is_some());
    }

    #[test]
    fn x_must_be_standalone() {
        assert!(find_expression("3x4").is_none(), "3x4 could be a label");
        assert!(find_expression("matrix 3 x 4").is_some());
    }

    #[test]
    fn longest_chain_wins() {
        // Two candidate chains; the 3-operand one is the expression.
        let e = find_expression("page 7 + 1, then 10 + 20 + 30").expect("expr");
        assert_eq!(e.to_string(), "10 + 20 + 30");
    }

    #[test]
    fn rewrite_prompt_embeds_question_and_two_shots() {
        let p = rewrite_prompt("If a box holds 12 eggs, how many in 4 boxes?");
        assert!(p.contains("7 + 5"));
        assert!(p.contains("240 * 3"));
        assert!(p.ends_with("E:"));
        assert!(p.contains("how many in 4 boxes?"));
    }

    #[test]
    fn parse_rewrite_reads_first_line_only() {
        let e = parse_rewrite(" 240 * 3\nQ: another question\nE: 1 + 1").expect("expr");
        assert_eq!(e.to_string(), "240 * 3");
    }

    #[test]
    fn parse_rewrite_discards_a_volunteered_sum() {
        let e = parse_rewrite(" 7 + 5 = 12").expect("expr");
        assert_eq!(e.to_string(), "7 + 5");
    }

    #[test]
    fn parse_rewrite_misses_on_garbage() {
        assert!(parse_rewrite("I cannot rewrite that.").is_none());
        assert!(parse_rewrite("").is_none());
    }
}
