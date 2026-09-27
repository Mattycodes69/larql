//! Trampoline tests. The `research` module runs under default features;
//! the `release` module runs under `--no-default-features`, the shape
//! tagged release binaries ship in.

use super::*;

fn args(tokens: &[&str]) -> Vec<String> {
    tokens.iter().map(|s| s.to_string()).collect()
}

#[test]
fn primary_verb_is_untouched() {
    let input = args(&["larql", "run", "gemma3-4b.vindex", "hello"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[test]
fn top_level_extract_is_untouched() {
    let input = args(&["larql", "extract", "google/gemma-3-4b-it", "-o", "out"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[test]
fn extract_index_alias_is_untouched() {
    // `extract-index` is a distinct top-level variant, not a legacy
    // research command — must not be rewritten or refused.
    let input = args(&["larql", "extract-index", "google/gemma-3-4b-it"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[test]
fn no_args_returns_unchanged() {
    let input = args(&["larql"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[test]
fn unknown_verb_is_passed_to_clap() {
    // `larql typo-command` is clap's to reject, not ours.
    let input = args(&["larql", "typo-command"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[test]
fn user_facing_shannon_subcommand_is_untouched() {
    let input = args(&["larql", "shannon", "score", "m", "--text", "x"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[test]
fn dec_bench_is_untouched() {
    // dec-bench ships in release binaries (the DEC drivers run it).
    let input = args(&["larql", "dec-bench", "replay", "--help"]);
    assert_eq!(prepare_argv(input.clone()), Ok(input));
}

#[cfg(feature = "research")]
mod research {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn legacy_research_verb_is_rewritten() {
        let input = args(&["larql", "walk", "--index", "x.vindex", "--predict"]);
        assert_eq!(
            rewrite_legacy_argv(input),
            args(&["larql", "dev", "walk", "--index", "x.vindex", "--predict"])
        );
    }

    #[test]
    fn legacy_research_flag_names_all_rewrite() {
        for name in LEGACY_DEV_NAMES {
            let out = rewrite_legacy_argv(args(&["larql", name, "--help"]));
            assert_eq!(out, args(&["larql", "dev", name, "--help"]));
        }
    }

    #[test]
    fn rewrite_preserves_argument_count_plus_one() {
        let input = args(&["larql", "walk", "--flag", "value"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out.len(), input.len() + 1);
    }

    /// Every legacy name must rewrite to a subcommand that ACTUALLY
    /// EXISTS. The rewrite test above passes just as happily when the
    /// target is gone; three dead entries (`extract-routes`, `ffn-bench`,
    /// `ffn-throughput`) survived behind it until 2026-08-22, turning a
    /// clean top-level error into clap's "did you mean" for a different
    /// command.
    #[test]
    fn every_legacy_name_maps_to_a_real_dev_subcommand() {
        let cli = crate::Cli::command();
        let dev = cli
            .get_subcommands()
            .find(|c| c.get_name() == "dev")
            .expect("`dev` subcommand exists");
        let live: Vec<&str> = dev.get_subcommands().map(|c| c.get_name()).collect();
        let dead: Vec<&&str> = LEGACY_DEV_NAMES
            .iter()
            .filter(|n| !live.contains(&**n))
            .collect();
        assert!(
            dead.is_empty(),
            "legacy names with no dev subcommand: {dead:?}"
        );
    }

    /// The refusal list a release build uses must name real research
    /// verbs, or a release binary would refuse a verb nobody has.
    #[test]
    fn research_refusal_lists_name_real_commands() {
        let cli = crate::Cli::command();
        for verb in RESEARCH_TOP_LEVEL {
            assert!(
                cli.find_subcommand(verb).is_some(),
                "RESEARCH_TOP_LEVEL names `{verb}`, which is not a subcommand"
            );
        }
        let shannon = cli.find_subcommand("shannon").expect("`shannon` exists");
        for sub in RESEARCH_SHANNON_SUBCOMMANDS {
            assert!(
                shannon.find_subcommand(sub).is_some(),
                "RESEARCH_SHANNON_SUBCOMMANDS names `{sub}`, which is not a subcommand"
            );
        }
    }
}

#[cfg(not(feature = "research"))]
mod release {
    use super::*;
    use clap::CommandFactory;

    fn refused(tokens: &[&str]) -> ResearchUnavailable {
        prepare_argv(args(tokens)).expect_err("research command must be refused")
    }

    #[test]
    fn legacy_research_names_are_refused_by_name() {
        for name in LEGACY_DEV_NAMES {
            assert_eq!(refused(&["larql", name, "--help"]).command, *name);
        }
    }

    #[test]
    fn gated_top_level_verbs_are_refused() {
        for verb in RESEARCH_TOP_LEVEL {
            assert_eq!(refused(&["larql", verb]).command, *verb);
        }
    }

    #[test]
    fn shannon_verify_is_refused() {
        assert_eq!(
            refused(&["larql", "shannon", "verify", "m"]).command,
            "shannon verify"
        );
    }

    #[test]
    fn refusal_names_the_missing_feature() {
        let message = refused(&["larql", "walk"]).to_string();
        assert!(message.contains("`larql walk`"), "{message}");
        assert!(message.contains("`research` cargo feature"), "{message}");
    }

    /// A release binary must not carry any of the verbs it refuses —
    /// otherwise the refusal would hide a working command.
    #[test]
    fn refused_verbs_are_absent_from_the_release_cli() {
        let cli = crate::Cli::command();
        for verb in RESEARCH_TOP_LEVEL.iter().chain(LEGACY_DEV_NAMES) {
            assert!(
                cli.find_subcommand(verb).is_none(),
                "`{verb}` is refused but still a subcommand"
            );
        }
        let shannon = cli.find_subcommand("shannon").expect("`shannon` exists");
        for sub in RESEARCH_SHANNON_SUBCOMMANDS {
            assert!(shannon.find_subcommand(sub).is_none(), "`shannon {sub}`");
        }
    }
}
