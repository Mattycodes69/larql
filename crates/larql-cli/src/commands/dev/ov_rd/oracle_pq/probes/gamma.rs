//! `--address-gamma-projected-group-probe`: the supervised group-bit
//! probes trained on the layer input after a gamma-alignment projection
//! toward later residual snapshots (plus random and learned low-rank
//! bridge controls).

use crate::commands::dev::ov_rd::gamma_address::{
    fit_gamma_projected_address_models, GammaProjectedAddressModel,
};
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_layer_input_hidden;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    rest_codes, rest_variants, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult,
    ProbeTargets,
};
use super::super::validate::{check_groups_all, parse_sorted, when};
use super::supervised::SgdSettings;

/// The learned low-rank bridge fit.
#[derive(Clone, Copy)]
struct LearnedBridge {
    epochs: usize,
    lr: f32,
    l2: f32,
    pca_iters: usize,
}

pub(in super::super) struct GammaProjectedProbe {
    enabled: bool,
    groups: Vec<usize>,
    layers: Vec<usize>,
    random_ranks: Vec<usize>,
    random_seeds: Vec<u64>,
    learned_ranks: Vec<usize>,
    learned: LearnedBridge,
    sgd: SgdSettings,
    models: HeadConfigMap<Vec<GammaProjectedAddressModel>>,
}

impl GammaProjectedProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_gamma_projected_groups)?;
        let layers = parse_sorted(&args.address_gamma_projected_layers)?;
        let random_ranks = parse_sorted(&args.address_gamma_random_ranks)?;
        let mut random_seeds = parse_sorted(&args.address_gamma_random_seeds)?
            .into_iter()
            .map(|seed| seed as u64)
            .collect::<Vec<_>>();
        random_seeds.sort_unstable();
        random_seeds.dedup();
        let learned_ranks = parse_sorted(&args.address_gamma_learned_ranks)?;
        let probe = Self {
            enabled: args.address_gamma_projected_group_probe,
            groups,
            layers,
            random_ranks,
            random_seeds,
            learned_ranks,
            learned: LearnedBridge {
                epochs: args.address_gamma_learned_epochs,
                lr: args.address_gamma_learned_lr,
                l2: args.address_gamma_learned_l2,
                pca_iters: args.address_gamma_learned_pca_iters,
            },
            sgd: SgdSettings::from_args(args),
            models: HeadConfigMap::new(),
        };
        if probe.enabled {
            probe.validate(targets)?;
        }
        Ok(probe)
    }

    fn validate(&self, targets: &ProbeTargets<'_>) -> ProbeResult {
        if self.groups.is_empty() {
            return Err("--address-gamma-projected-group-probe requires at least one --address-gamma-projected-groups value".into());
        }
        if self.layers.is_empty() && self.random_ranks.is_empty() && self.learned_ranks.is_empty() {
            return Err("--address-gamma-projected-layers, --address-gamma-random-ranks, or --address-gamma-learned-ranks must include at least one value".into());
        }
        if !self.learned_ranks.is_empty() && self.layers.is_empty() {
            return Err(
                "--address-gamma-learned-ranks requires at least one --address-gamma-projected-layers value"
                    .into(),
            );
        }
        for &layer in &self.layers {
            if layer >= targets.num_layers {
                return Err(format!(
                    "--address-gamma-projected-layers includes layer {layer}, but the model has only {} layers",
                    targets.num_layers
                )
                .into());
            }
        }
        for head in targets.heads {
            for &layer in &self.layers {
                if layer < head.layer {
                    return Err(format!(
                        "--address-gamma-projected-layers includes post-L{layer}, before target L{}H{}",
                        head.layer, head.head
                    )
                    .into());
                }
            }
        }
        let rank_range = 1..=targets.hidden_size;
        for &rank in &self.random_ranks {
            if !rank_range.contains(&rank) {
                return Err(format!(
                    "--address-gamma-random-ranks includes rank {rank}, expected 1..={}",
                    targets.hidden_size
                )
                .into());
            }
        }
        if !self.random_ranks.is_empty() && self.random_seeds.is_empty() {
            return Err(
                "--address-gamma-random-seeds must include at least one seed when random ranks are enabled"
                    .into(),
            );
        }
        for &rank in &self.learned_ranks {
            if !rank_range.contains(&rank) {
                return Err(format!(
                    "--address-gamma-learned-ranks includes rank {rank}, expected 1..={}",
                    targets.hidden_size
                )
                .into());
            }
        }
        if self.learned.epochs == 0 {
            return Err("--address-gamma-learned-epochs must be greater than zero".into());
        }
        if self.learned.lr <= 0.0 {
            return Err("--address-gamma-learned-lr must be greater than zero".into());
        }
        if self.learned.l2 < 0.0 {
            return Err("--address-gamma-learned-l2 must be non-negative".into());
        }
        if self.learned.pca_iters == 0 {
            return Err("--address-gamma-learned-pca-iters must be greater than zero".into());
        }
        check_groups_all(
            "--address-gamma-projected-groups",
            &self.groups,
            targets.configs,
        )
    }
}

impl AddressProbe for GammaProjectedProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-gamma-projected-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting gamma-projected supervised group address probes for groups {:?} (post_layers={:?}, random_ranks={:?}, random_seeds={:?}, learned_ranks={:?}, learned_epochs={}, learned_lr={}, learned_l2={}, learned_pca_iters={}, epochs={}, lr={}, l2={})",
            self.groups,
            self.layers,
            self.random_ranks,
            self.random_seeds,
            self.learned_ranks,
            self.learned.epochs,
            self.learned.lr,
            self.learned.l2,
            self.learned.pca_iters,
            self.sgd.epochs,
            self.sgd.lr,
            self.sgd.l2
        );
        self.models = fit_gamma_projected_address_models(
            ctx.weights,
            ctx.index,
            ctx.tokenizer,
            ctx.fit_prompts,
            ctx.heads,
            ctx.bases,
            ctx.means,
            ctx.pca_bases,
            ctx.codebooks,
            &self.groups,
            &self.layers,
            &self.random_ranks,
            &self.random_seeds,
            &self.learned_ranks,
            self.learned.epochs,
            self.learned.lr,
            self.learned.l2,
            self.learned.pca_iters,
            self.sgd.epochs,
            self.sgd.lr,
            self.sgd.l2,
        )?;
        Ok(())
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("gamma-projected group probe")?;
        let gamma_models = ctx.model(&self.models, "gamma-projected group probe models")?;
        let group_majority = ctx.majority("gamma-projected group probe")?;
        let layer_input =
            capture_layer_input_hidden(ctx.weights, ctx.token_ids, ctx.index, ctx.head.layer)?;
        for gamma_model in gamma_models {
            let projected_input = gamma_model.project_layer_input(&layer_input)?;
            let selected_group_keys = gamma_model.selected_group_keys();
            for (probe_name, use_oracle_rest) in rest_variants(&gamma_model.name, &self.groups) {
                let predicted_codes_by_position = ctx
                    .oracle_codes_by_position
                    .iter()
                    .enumerate()
                    .map(|(pos, oracle_codes)| {
                        let base_codes = rest_codes(use_oracle_rest, oracle_codes, group_majority);
                        gamma_model.supervised.predict_selected_groups(
                            &projected_input,
                            pos,
                            base_codes,
                        )
                    })
                    .collect::<Vec<_>>();
                ctx.evaluate_probe(
                    mode_d_table,
                    &predicted_codes_by_position,
                    &probe_name,
                    &selected_group_keys,
                )?;
            }
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        let enabled = self.enabled;
        report.address_gamma_projected_group_probe = enabled;
        report.address_gamma_projected_groups = when(enabled, self.groups.clone());
        report.address_gamma_projected_layers = when(enabled, self.layers.clone());
        report.address_gamma_random_ranks = when(enabled, self.random_ranks.clone());
        report.address_gamma_random_seeds = when(enabled, self.random_seeds.clone());
        report.address_gamma_learned_ranks = when(enabled, self.learned_ranks.clone());
        report.address_gamma_learned_epochs = when(enabled, self.learned.epochs);
        report.address_gamma_learned_lr = when(enabled, self.learned.lr);
        report.address_gamma_learned_l2 = when(enabled, self.learned.l2);
        report.address_gamma_learned_pca_iters = when(enabled, self.learned.pca_iters);
    }
}
