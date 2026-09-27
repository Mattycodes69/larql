//! vindex — open a VINDEX3 artifact and interrogate it.
//!
//! Deliberately small: the format-native verbs only, each answering
//! from the container's own declarations, each speaking `--json`. The
//! text renderings below are projections of the same facts object the
//! JSON emits — one result, two views, and the web Explorer's designed
//! panels are the third.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod render;
use render::*;

mod update;

#[derive(Parser)]
#[command(
    name = "vindex",
    about = "Format-native VINDEX3 tooling: plan, encode, inspect, verify, compile representations and export.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Emit the structured result instead of text.
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Encode a checkpoint into a VINDEX3 container.
    ///
    /// An `hf://org/name[@revision]` argument is read from Hugging Face
    /// over byte ranges — the canonical checkpoint never needs to exist
    /// as a complete local download.
    Encode {
        /// Checkpoint directories, saved inventories, or `hf://` repos.
        #[arg(required = true)]
        artifacts: Vec<PathBuf>,
        /// Container directory to write.
        #[arg(long)]
        output: PathBuf,
        /// Admit on text generation's execution closure instead of
        /// whole-model completeness. The container written is identical
        /// either way; only the gate changes.
        #[arg(long)]
        text_only: bool,
    },
    /// What VINDEX understands about a model, before moving its weights.
    ///
    /// Reads configuration and safetensors headers only. A bring-up
    /// instrument: it answers what VINDEX still needs to understand, not
    /// whether you can use the model.
    Plan {
        /// Checkpoint directories, saved inventories, or `hf://` repos.
        #[arg(required = true)]
        artifacts: Vec<PathBuf>,
    },
    /// The container, reconstructed from itself: identity, census, coherence.
    Inspect { container: PathBuf },
    /// One logical object, in full — identity, bindings, representations, tensor-table head.
    Describe {
        container: PathBuf,
        /// A logical object id, or an unambiguous suffix of one.
        address: String,
        /// How many tensor-table rows to show per representation.
        #[arg(long, default_value_t = 8)]
        values: usize,
        /// Decode and print the first values of this tensor (name or
        /// suffix) — the numbers themselves, from the canonical bytes.
        #[arg(long)]
        peek: Option<String>,
    },
    /// The physical directory: what exists as bytes, with recorded fidelity.
    Representations { container: PathBuf },
    /// Every layer's token-mixer programme, from the operation plan.
    Layers { container: PathBuf },
    /// One object under two of the container's representations, decoded and
    /// compared value by value — the error derived, never asserted.
    Diff {
        container: PathBuf,
        /// First encoding (e.g. F32, BF16).
        a: String,
        /// Second encoding (e.g. NVFP4).
        b: String,
        /// A logical object id, or an unambiguous suffix of one.
        address: String,
        /// How many per-value rows to show.
        #[arg(long, default_value_t = 8)]
        values: usize,
        /// Show values from this tensor (name or suffix) instead of the
        /// tensor with the largest error.
        #[arg(long)]
        tensor: Option<String>,
    },
    /// Compile a representation through the reference compiler into a new
    /// container beside the original. Nothing is destroyed.
    Represent {
        container: PathBuf,
        /// Where to write the new container.
        out: PathBuf,
        /// Target encoding.
        #[arg(long, default_value = "NVFP4")]
        encoding: String,
    },
    /// Compile the selected representation to a GGUF for an independent
    /// runtime, verified against the plan before the command returns.
    Export {
        container: PathBuf,
        /// The .gguf file to write.
        out: PathBuf,
    },
    /// Bits per weight — derived from stored bytes over tensor elements, never asserted.
    Precision {
        container: PathBuf,
        /// The precision map, seen: bits per layer × semantic role, from the
        /// representation each object would execute.
        #[arg(long)]
        matrix: bool,
    },
    /// The container against its own recorded hashes, re-derived from the artifact alone.
    Verify { container: PathBuf },
    /// Install the latest release of this tool. Only ever runs when you ask:
    /// no verb checks for updates on its own, and nothing phones home.
    Update {
        /// Report whether a newer release exists, without installing.
        #[arg(long)]
        check: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Command::Update { check } = &cli.command {
        return match update::run(*check) {
            Ok(line) => {
                println!("{line}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("vindex: {e}");
                ExitCode::from(2)
            }
        };
    }
    let result = match &cli.command {
        Command::Encode {
            artifacts,
            output,
            text_only,
        } => vindex_cli::encode_facts(artifacts, output, *text_only),
        Command::Plan { artifacts } => vindex_cli::plan_facts(artifacts),
        Command::Inspect { container } => vindex_cli::inspect_facts(container),
        Command::Describe {
            container,
            address,
            values,
            peek,
        } => vindex_cli::describe_facts(container, address, *values, peek.as_deref()),
        Command::Representations { container } => vindex_cli::representations_facts(container),
        Command::Layers { container } => vindex_cli::layers_facts(container),
        Command::Diff {
            container,
            a,
            b,
            address,
            values,
            tensor,
        } => vindex_cli::diff_facts(container, a, b, address, *values, tensor.as_deref()),
        Command::Represent {
            container,
            out,
            encoding,
        } => vindex_cli::represent_facts(container, out, encoding),
        Command::Precision { container, matrix } => {
            if *matrix {
                vindex_cli::precision_matrix_facts(container)
            } else {
                vindex_cli::precision_facts(container)
            }
        }
        Command::Verify { container } => vindex_cli::verify_facts(container),
        Command::Export { container, out } => vindex_cli::export_facts(container, out),
        Command::Update { .. } => unreachable!("handled above"),
    };
    match result {
        Ok(v) => {
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            } else {
                match &cli.command {
                    Command::Encode { .. } => render_encode(&v),
                    Command::Plan { .. } => render_plan(&v),
                    Command::Inspect { .. } => render_inspect(&v),
                    Command::Export { .. } => render_export(&v),
                    Command::Describe { .. } => render_describe(&v),
                    Command::Representations { .. } => render_representations(&v),
                    Command::Layers { .. } => render_layers(&v),
                    Command::Diff { .. } => render_diff(&v),
                    Command::Represent { .. } => render_represent(&v),
                    Command::Precision { matrix, .. } => {
                        if *matrix {
                            render_precision_matrix(&v)
                        } else {
                            render_precision(&v)
                        }
                    }
                    Command::Verify { .. } => render_verify(&v),
                    Command::Update { .. } => unreachable!("handled above"),
                }
            }
            if let Some(false) = v["verified"].as_bool() {
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("vindex: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod documentation_tests {
    use clap::CommandFactory;
    use larql_vindex::format::generation::{
        ContainerGeneration, DEFAULT_EXTRACTION_GENERATION, V3_CURRENT_SCHEMA,
    };
    use larql_vindex::format::vindex3::{
        graph::GRAPH_SCHEMA,
        plan::{PLANNER_SEMANTICS_VERSION, PLAN_SCHEMA},
    };

    #[test]
    fn current_documentation_facts_match_code_and_clap() {
        let facts: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/generated/current-facts.json"))
                .unwrap();
        assert_eq!(facts["vindex_cli_version"], env!("CARGO_PKG_VERSION"));
        let constants = &facts["constants"];
        assert_eq!(constants["V3_CURRENT_SCHEMA"], V3_CURRENT_SCHEMA);
        assert_eq!(constants["GRAPH_SCHEMA"], GRAPH_SCHEMA);
        assert_eq!(constants["PLAN_SCHEMA"], PLAN_SCHEMA);
        assert_eq!(
            constants["PLANNER_SEMANTICS_VERSION"],
            PLANNER_SEMANTICS_VERSION
        );
        let generation = match DEFAULT_EXTRACTION_GENERATION {
            ContainerGeneration::V2 => "V2",
            ContainerGeneration::V3 => "V3",
        };
        assert_eq!(constants["DEFAULT_EXTRACTION_GENERATION"], generation);
        let command = super::Cli::command();
        let mut names: Vec<_> = command.get_subcommands().map(|c| c.get_name()).collect();
        names.sort_unstable();
        assert_eq!(facts["commands"]["vindex"], serde_json::json!(names));
    }
}
