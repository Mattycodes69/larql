//! lexer.rs — strings, escapes, operators, comments, error chars

use super::*;

#[test]
fn string_escapes_decode_through_parser() {
    let stmt = ok(r#"WALK "line1\nline2\ttab\r\\back\0null \"q\" 'x'";"#);
    match stmt {
        Statement::Walk { prompt, .. } => {
            assert!(prompt.contains('\n'));
            assert!(prompt.contains('\t'));
            assert!(prompt.contains('\\'));
            assert!(prompt.contains('\0'));
            assert!(prompt.contains('"'));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn single_quoted_string_with_escaped_apostrophe() {
    let stmt = ok(r"WALK 'it\'s fine';");
    match stmt {
        Statement::Walk { prompt, .. } => assert_eq!(prompt, "it's fine"),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn unknown_escape_passes_through_unchanged() {
    // `\q` → `q` (the `other => other as char` arm in read_quoted).
    let stmt = ok(r#"WALK "a\qb";"#);
    match stmt {
        Statement::Walk { prompt, .. } => assert_eq!(prompt, "aqb"),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn unterminated_string_is_error() {
    err(r#"WALK "no closing quote"#);
}

#[test]
fn unterminated_after_backslash_is_error() {
    err(r#"WALK "abc\"#);
}

#[test]
fn line_comments_are_skipped() {
    let stmt = ok("-- a comment\nSTATS; -- trailing comment");
    assert!(matches!(stmt, Statement::Stats { .. }));
}

#[test]
fn pipe_operator_chains_statements() {
    let stmt = ok(r#"WALK "x" TOP 3 |> SELECT * FROM EDGES;"#);
    assert!(matches!(stmt, Statement::Pipe { .. }));
}

#[test]
fn incomplete_pipe_is_error() {
    // '|' not followed by '>' (lexer LexError branch).
    err(r#"WALK "x" | SELECT * FROM EDGES;"#);
}

#[test]
fn bang_without_eq_is_error() {
    // '!' not followed by '=' (lexer LexError branch).
    err("SELECT * FROM EDGES WHERE a ! b;");
}

#[test]
fn unexpected_character_is_error() {
    // '@' is not a valid token start.
    err("SELECT @ FROM EDGES;");
}

#[test]
fn all_comparison_operators_parse() {
    // Eq / Neq / Gt / Lt / Gte / Lte plus LIKE and IN exercise parse_compare_op.
    ok("SELECT * FROM EDGES WHERE a = 1;");
    ok("SELECT * FROM EDGES WHERE a != 1;");
    ok("SELECT * FROM EDGES WHERE a > 1;");
    ok("SELECT * FROM EDGES WHERE a < 1;");
    ok("SELECT * FROM EDGES WHERE a >= 1;");
    ok("SELECT * FROM EDGES WHERE a <= 1;");
    ok(r#"SELECT * FROM EDGES WHERE a LIKE "%x%";"#);
    ok("SELECT * FROM EDGES WHERE a IN (1, 2, 3);");
}

#[test]
fn lex_error_display_is_rendered() {
    // lexer.rs 637-639: LexError::Display. `parse` surfaces lex errors as a
    // boxed std::error::Error; formatting it routes through Display.
    let e = parse(r#"WALK "unterminated"#).unwrap_err();
    let msg = format!("{e}");
    assert!(
        msg.contains("Lex error"),
        "expected lexer Display prefix, got {msg:?}"
    );
}
