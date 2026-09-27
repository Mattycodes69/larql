//! Integration coverage for the LQL lexer and parser helper/lifecycle paths.
//!
//! These exercise the PURE parsing surface only — no vindex/model needed.
//! Everything is driven through the public `larql_lql::parser::parse` entry
//! point (the lexer is `pub(crate)`, so it is reached transitively).
//!
//! Goals (≥90% line coverage):
//!   * lexer.rs — every keyword's `as_field_name` arm (150-252), the float
//!     fractional-digit loop (588), and `LexError::Display` (637-639).
//!   * parser/helpers.rs — optional-clause and error branches.
//!   * parser/lifecycle.rs — USE / COMPILE / DIFF / COMPACT variants + errors.

use larql_lql::ast::Statement;
use larql_lql::parser::parse;

fn ok(sql: &str) -> Statement {
    match parse(sql) {
        Ok(stmt) => stmt,
        Err(e) => panic!("expected Ok parsing {sql:?}, got Err: {e}"),
    }
}

fn err(sql: &str) {
    let r = parse(sql);
    assert!(
        r.is_err(),
        "expected Err parsing {sql:?}, got Ok: {:?}",
        r.ok()
    );
}

// lexer.rs 150-252 — `Keyword::as_field_name` for EVERY keyword variant.
//
// `parse_field` (helpers.rs 203-207) maps `Token::Keyword(kw)` to
// `kw.as_field_name()`. Driving `SELECT <kw> FROM EDGES` therefore exercises
// one `as_field_name` match arm per keyword. parse_field reads a single field
// (no trailing comma) then the FROM clause consumes the next `FROM`, so even
// keywords like FROM/SELECT/EDGES work as a single field name here.

/// Every keyword as it is spelled for the lexer's case-insensitive matcher.
/// Order/spelling mirrors `Keyword::from_str` in lexer.rs.
const ALL_KEYWORDS: &[&str] = &[
    "EXTRACT",
    "COMPILE",
    "DIFF",
    "USE",
    "WALK",
    "SELECT",
    "DESCRIBE",
    "EXPLAIN",
    "INSERT",
    "DELETE",
    "UPDATE",
    "MERGE",
    "SHOW",
    "STATS",
    "FROM",
    "INTO",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "LIKE",
    "BETWEEN",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "TOP",
    "LAYERS",
    "MODE",
    "COMPARE",
    "AT",
    "LAYER",
    "CONFIDENCE",
    "MODEL",
    "EDGES",
    "RELATION",
    "RELATIONS",
    "ENTITIES",
    "FEATURES",
    "MODELS",
    "FORMAT",
    "COMPONENTS",
    "ON",
    "CONFLICT",
    "KEEP_SOURCE",
    "KEEP_TARGET",
    "HIGHEST_CONFIDENCE",
    "LAST_WINS",
    "FAIL",
    "FOR",
    "SET",
    "VALUES",
    "CURRENT",
    "WITH",
    "EXAMPLES",
    "ONLY",
    "VERBOSE",
    "RANGE",
    "ALL",
    "NEAREST",
    "TO",
    "PURE",
    "HYBRID",
    "DENSE",
    "SAFETENSORS",
    "GGUF",
    "AUTO_EXTRACT",
    "FFN_GATE",
    "FFN_DOWN",
    "FFN_UP",
    "EMBEDDINGS",
    "ATTN_OV",
    "ATTN_QK",
    "INFER",
    "SYNTAX",
    "KNOWLEDGE",
    "OUTPUT",
    "WEIGHTS",
    "INFERENCE",
    "BEGIN",
    "SAVE",
    "APPLY",
    "REMOVE",
    "PATCH",
    "PATCHES",
    "REMOTE",
    "TRACE",
    "DECOMPOSE",
    "POSITIONS",
    "BRIEF",
    "RAW",
    "ATTENTION",
    "ALPHA",
    "KNN",
    "COMPOSE",
    "REBALANCE",
    "FLOOR",
    "CEILING",
    "MAX",
    "UNTIL",
    "CONVERGED",
    "COMPACT",
    "STATUS",
];

// lexer.rs — numeric literals

// lexer.rs — strings, escapes, operators, comments, error chars

// parser/helpers.rs — value list, field list, compare-op + token errors

// parser/lifecycle.rs — USE / COMPILE / DIFF / EXTRACT / COMPACT

// Cross-cutting: statement dispatch, pipe, trailing-token, empty input

mod lexer_rs_strings_escapes_operators_comme;
mod parse_walk_mode_parse_output_format_pars;
mod parser_helpers_rs_value_list_field_list;
mod parser_lifecycle_rs_use_compile_diff_ext;
mod small_assertion_helpers;
