use super::*;
use crate::{Cli, Commands};
use clap::Parser;

#[test]
fn serve_forwards_explicit_backend_and_rejects_unknown_names() {
    for backend in ["cpu", "metal"] {
        let cli = Cli::try_parse_from(["larql", "serve", "--v3-backend", backend]).unwrap();
        let Commands::Serve(args) = cli.command else {
            panic!("expected serve")
        };
        let command = serve_command_args(&args).unwrap();
        assert!(command
            .windows(2)
            .any(|pair| pair == ["--v3-backend", backend]));
    }
    assert!(Cli::try_parse_from(["larql", "serve", "--v3-backend", "typo"]).is_err());
}
