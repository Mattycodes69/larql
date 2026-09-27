//! Shared flag parsing and range checks for the probe settings.

use crate::commands::dev::ov_rd::input::parse_usize_list;
use crate::commands::dev::ov_rd::types::PqConfig;

use super::probe::ProbeResult;

/// A comma list of indices, sorted and de-duplicated.
pub(super) fn parse_sorted(spec: &str) -> ProbeResult<Vec<usize>> {
    let mut values = parse_usize_list(spec)?;
    values.sort_unstable();
    values.dedup();
    Ok(values)
}

/// Number of codes a PQ group can take.
pub(super) fn levels(config: &PqConfig) -> usize {
    1usize << config.bits_per_group
}

/// Refuse a group index the config does not have.
pub(super) fn check_groups(flag: &str, groups: &[usize], config: &PqConfig) -> ProbeResult {
    for &group in groups {
        if group >= config.groups {
            return Err(format!(
                "{flag} includes group {group}, but config {:?} has only {} groups",
                config, config.groups
            )
            .into());
        }
    }
    Ok(())
}

/// [`check_groups`] across every config.
pub(super) fn check_groups_all(flag: &str, groups: &[usize], configs: &[PqConfig]) -> ProbeResult {
    for config in configs {
        check_groups(flag, groups, config)?;
    }
    Ok(())
}

/// Refuse a code the config's group width cannot express.
pub(super) fn check_codes(flag: &str, codes: &[usize], config: &PqConfig) -> ProbeResult {
    let levels = levels(config);
    for &code in codes {
        if code >= levels {
            return Err(format!(
                "{flag} includes code {code}, but config {:?} has only {levels} levels",
                config
            )
            .into());
        }
    }
    Ok(())
}

/// A setting as the report records it: its value when the probe is
/// enabled, the type's zero value when it is not.
pub(super) fn when<T: Default>(enabled: bool, value: T) -> T {
    if enabled {
        value
    } else {
        T::default()
    }
}
