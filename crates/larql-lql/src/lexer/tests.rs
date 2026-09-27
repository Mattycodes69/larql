use super::*;

// ── Basic tokenisation ──

#[test]
fn walk_simple() {
    let mut lex = Lexer::new(r#"WALK "The capital of France is" TOP 5;"#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Walk)));
    assert!(matches!(tokens[1], Token::StringLit(ref s) if s == "The capital of France is"));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::Top)));
    assert!(matches!(tokens[3], Token::IntegerLit(5)));
    assert!(matches!(tokens[4], Token::Semicolon));
}

#[test]
fn use_vindex() {
    let mut lex = Lexer::new(r#"USE "gemma3-4b.vindex";"#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Use)));
    assert!(matches!(tokens[1], Token::StringLit(ref s) if s == "gemma3-4b.vindex"));
}

#[test]
fn select_with_conditions() {
    let mut lex =
        Lexer::new(r#"SELECT entity, relation FROM EDGES WHERE entity = "France" LIMIT 10;"#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Select)));
    assert!(matches!(tokens[1], Token::Ident(ref s) if s == "entity"));
    assert!(matches!(tokens[2], Token::Comma));
    assert!(matches!(tokens[3], Token::Keyword(Keyword::Relation)));
}

// ── Comments ──

#[test]
fn comment_skipping() {
    let mut lex = Lexer::new("-- this is a comment\nSTATS;");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Stats)));
}

#[test]
fn multiple_comments() {
    let input = "-- first comment\n-- second comment\nSTATS;\n-- trailing";
    let mut lex = Lexer::new(input);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Stats)));
    assert!(matches!(tokens[1], Token::Semicolon));
    assert!(matches!(tokens[2], Token::Eof));
}

#[test]
fn inline_comment_after_statement() {
    let mut lex = Lexer::new("STATS; -- inline comment");
    let tokens = lex.tokenise().unwrap();
    assert_eq!(tokens.len(), 3); // STATS, ;, EOF
}

// ── Numbers ──

#[test]
fn integer_literal() {
    let mut lex = Lexer::new("42");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::IntegerLit(42)));
}

#[test]
fn float_literal() {
    let mut lex = Lexer::new("0.89");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::NumberLit(n) if (n - 0.89).abs() < 0.001));
}

#[test]
fn negative_number_is_dash_plus_int() {
    let mut lex = Lexer::new("-5");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Dash));
    assert!(matches!(tokens[1], Token::IntegerLit(5)));
}

#[test]
fn range_with_dash() {
    let mut lex = Lexer::new("0-33");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::IntegerLit(0)));
    assert!(matches!(tokens[1], Token::Dash));
    assert!(matches!(tokens[2], Token::IntegerLit(33)));
}

// ── Strings ──

#[test]
fn double_quoted_string() {
    let mut lex = Lexer::new(r#""hello world""#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "hello world"));
}

#[test]
fn single_quoted_string() {
    let mut lex = Lexer::new("'hello world'");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "hello world"));
}

#[test]
fn string_with_escaped_quote_decodes() {
    let mut lex = Lexer::new(r#""hello \"world\"""#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == r#"hello "world""#));
}

#[test]
fn string_with_escaped_backslash_decodes() {
    let mut lex = Lexer::new(r#""C:\\path""#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == r"C:\path"));
}

#[test]
fn string_with_control_escapes_decodes() {
    let mut lex = Lexer::new(r#""a\nb\tc\rd\0e""#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "a\nb\tc\rd\0e"));
}

#[test]
fn single_quoted_string_with_escape() {
    let mut lex = Lexer::new(r"'it\'s'");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "it's"));
}

#[test]
fn unknown_escape_passes_through() {
    let mut lex = Lexer::new(r#""\x""#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "x"));
}

#[test]
fn unterminated_string_after_backslash() {
    let mut lex = Lexer::new(r#""abc\"#);
    assert!(lex.tokenise().is_err());
}

#[test]
fn keyword_as_field_name_covers_compact_and_status() {
    // Regression: prior implementation routed Compact / Status through an
    // `unreachable!()` and panicked when they appeared as identifiers in
    // WHERE / SET / ORDER BY clauses.
    assert_eq!(Keyword::Compact.as_field_name(), "compact");
    assert_eq!(Keyword::Status.as_field_name(), "status");
}

#[test]
fn unterminated_string_error() {
    let mut lex = Lexer::new(r#""unterminated"#);
    assert!(lex.tokenise().is_err());
}

#[test]
fn empty_string() {
    let mut lex = Lexer::new(r#""""#);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::StringLit(ref s) if s.is_empty()));
}

// ── Operators ──

#[test]
fn comparison_operators() {
    let mut lex = Lexer::new("= != > < >= <=");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Eq));
    assert!(matches!(tokens[1], Token::Neq));
    assert!(matches!(tokens[2], Token::Gt));
    assert!(matches!(tokens[3], Token::Lt));
    assert!(matches!(tokens[4], Token::Gte));
    assert!(matches!(tokens[5], Token::Lte));
}

#[test]
fn pipe_operator() {
    let mut lex = Lexer::new("|>");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Pipe));
}

#[test]
fn all_punctuation() {
    let mut lex = Lexer::new("* , ; ( ) .");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Star));
    assert!(matches!(tokens[1], Token::Comma));
    assert!(matches!(tokens[2], Token::Semicolon));
    assert!(matches!(tokens[3], Token::LParen));
    assert!(matches!(tokens[4], Token::RParen));
    assert!(matches!(tokens[5], Token::Dot));
}

// ── Keywords ──

#[test]
fn case_insensitive_keywords() {
    let mut lex = Lexer::new("walk WALK Walk wAlK");
    let tokens = lex.tokenise().unwrap();
    for (i, tok) in tokens.iter().take(4).enumerate() {
        assert!(
            matches!(tok, Token::Keyword(Keyword::Walk)),
            "token {i} should be Walk keyword"
        );
    }
}

#[test]
fn all_lifecycle_keywords() {
    let mut lex = Lexer::new("EXTRACT COMPILE DIFF USE");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Extract)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::Compile)));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::Diff)));
    assert!(matches!(tokens[3], Token::Keyword(Keyword::Use)));
}

#[test]
fn all_query_keywords() {
    let mut lex = Lexer::new("WALK SELECT DESCRIBE EXPLAIN");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Walk)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::Select)));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::Describe)));
    assert!(matches!(tokens[3], Token::Keyword(Keyword::Explain)));
}

#[test]
fn all_mutation_keywords() {
    let mut lex = Lexer::new("INSERT DELETE UPDATE MERGE");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Insert)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::Delete)));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::Update)));
    assert!(matches!(tokens[3], Token::Keyword(Keyword::Merge)));
}

#[test]
fn component_keywords() {
    let mut lex = Lexer::new("FFN_GATE FFN_DOWN FFN_UP EMBEDDINGS ATTN_OV ATTN_QK");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::FfnGate)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::FfnDown)));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::FfnUp)));
    assert!(matches!(tokens[3], Token::Keyword(Keyword::Embeddings)));
    assert!(matches!(tokens[4], Token::Keyword(Keyword::AttnOv)));
    assert!(matches!(tokens[5], Token::Keyword(Keyword::AttnQk)));
}

#[test]
fn mode_keywords() {
    let mut lex = Lexer::new("HYBRID PURE DENSE");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Hybrid)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::Pure)));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::Dense)));
}

#[test]
fn conflict_strategy_keywords() {
    let mut lex = Lexer::new("KEEP_SOURCE KEEP_TARGET HIGHEST_CONFIDENCE");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::KeepSource)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::KeepTarget)));
    assert!(matches!(
        tokens[2],
        Token::Keyword(Keyword::HighestConfidence)
    ));
}

#[test]
fn format_keywords() {
    let mut lex = Lexer::new("SAFETENSORS GGUF");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Safetensors)));
    assert!(matches!(tokens[1], Token::Keyword(Keyword::Gguf)));
}

// ── Identifiers ──

#[test]
fn unknown_word_is_ident() {
    let mut lex = Lexer::new("my_column foobar");
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Ident(ref s) if s == "my_column"));
    assert!(matches!(tokens[1], Token::Ident(ref s) if s == "foobar"));
}

// ── Error cases ──

#[test]
fn unexpected_character_error() {
    let mut lex = Lexer::new("@");
    assert!(lex.tokenise().is_err());
}

#[test]
fn incomplete_pipe_error() {
    let mut lex = Lexer::new("|x");
    assert!(lex.tokenise().is_err());
}

#[test]
fn incomplete_bang_error() {
    let mut lex = Lexer::new("!x");
    assert!(lex.tokenise().is_err());
}

// ── Empty / whitespace ──

#[test]
fn empty_input() {
    let mut lex = Lexer::new("");
    let tokens = lex.tokenise().unwrap();
    assert_eq!(tokens.len(), 1);
    assert!(matches!(tokens[0], Token::Eof));
}

#[test]
fn whitespace_only() {
    let mut lex = Lexer::new("   \n\t  \n  ");
    let tokens = lex.tokenise().unwrap();
    assert_eq!(tokens.len(), 1);
    assert!(matches!(tokens[0], Token::Eof));
}

// ── Full statement tokenisation ──

#[test]
fn extract_statement_tokens() {
    let input = r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "out.vindex" COMPONENTS FFN_GATE, FFN_DOWN LAYERS 0-33;"#;
    let mut lex = Lexer::new(input);
    let tokens = lex.tokenise().unwrap();
    // Count non-EOF tokens
    let count = tokens.iter().filter(|t| !matches!(t, Token::Eof)).count();
    assert!(count >= 12, "expected at least 12 tokens, got {count}");
}

#[test]
fn insert_statement_tokens() {
    let input =
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "lives-in", "London");"#;
    let mut lex = Lexer::new(input);
    let tokens = lex.tokenise().unwrap();
    // INSERT INTO EDGES ( entity , relation , target ) VALUES ( "John" , "lives-in" , "London" ) ;
    // =19 tokens: INSERT INTO EDGES ( entity , relation , target ) VALUES ( str , str , str ) ;
    let count = tokens.iter().filter(|t| !matches!(t, Token::Eof)).count();
    assert_eq!(count, 19);
}

#[test]
fn multiline_statement_tokens() {
    let input = "SELECT *\n  FROM EDGES\n  WHERE layer = 26\n  LIMIT 5;";
    let mut lex = Lexer::new(input);
    let tokens = lex.tokenise().unwrap();
    assert!(matches!(tokens[0], Token::Keyword(Keyword::Select)));
    assert!(matches!(tokens[1], Token::Star));
    assert!(matches!(tokens[2], Token::Keyword(Keyword::From)));
}
