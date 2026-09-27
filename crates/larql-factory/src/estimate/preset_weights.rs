//! Which coarse [`ComponentBytes`] fields sum to which preset.
//!
//! Coarser than the CLI's byte-exact `slice_cmd::Part` sets
//! (Gate/DownMeta/Norms/Tokenizer/Manifest/Labels are small metadata this
//! module doesn't model separately — folded into "close enough" or omitted,
//! never invented as their own byte count). The preset *names* are the
//! shared [`SlicePreset`] vocabulary, so a new preset fails to compile here
//! until it is given a meaning.

use larql_vindex_spec::{SlicePreset, UNSLICED_PRESET};

use super::bytes::ComponentBytes;

/// One coarse weight component. Deliberately smaller than
/// `slice_cmd::Part` — this module estimates order-of-magnitude bytes,
/// not an exact manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    Embed,
    Attn,
    Ffn,
    LmHead,
    Router,
}

/// Every component: the unsliced output and the `all` slice.
const EVERY_COMPONENT: &[Component] = &[
    Component::Embed,
    Component::Attn,
    Component::Ffn,
    Component::LmHead,
    Component::Router,
];

/// Components a slice preset includes.
fn slice_components(preset: SlicePreset) -> &'static [Component] {
    use Component::*;
    match preset {
        SlicePreset::All => EVERY_COMPONENT,
        SlicePreset::Client => &[Embed, Attn],
        SlicePreset::Attention => &[Attn],
        SlicePreset::Embed => &[Embed],
        SlicePreset::Server => &[Embed, Ffn],
        // Browse carries gate vectors + down_meta (compact per-feature
        // metadata), not full FFN weight blocks — approximated as
        // embed-only, a deliberate underestimate of the small
        // gate/down_meta contribution rather than double-counting FFN.
        SlicePreset::Browse => &[Embed],
        SlicePreset::Router => &[Router],
        SlicePreset::ExpertServer => &[Embed, Ffn, Router],
    }
}

/// Components a recipe output named `preset` includes, or `None` when the
/// name is neither the unsliced output nor a slice preset.
fn preset_components(preset: &str) -> Option<&'static [Component]> {
    if preset.eq_ignore_ascii_case(UNSLICED_PRESET) {
        return Some(EVERY_COMPONENT);
    }
    preset.parse().ok().map(slice_components)
}

/// Sum the components `preset` includes.
///
/// `None` for an unrecognised preset: an estimate for a name nothing can
/// build is unknown, not zero.
pub fn estimate_preset_bytes(preset: &str, components: &ComponentBytes) -> Option<u64> {
    let included = preset_components(preset)?;
    let total = included
        .iter()
        .map(|c| match c {
            Component::Embed => components.embed,
            Component::Attn => components.attn,
            Component::Ffn => components.ffn,
            Component::LmHead => components.lm_head,
            Component::Router => components.router,
        })
        .sum();
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_components() -> ComponentBytes {
        ComponentBytes {
            embed: 100,
            attn: 10,
            ffn: 1000,
            lm_head: 100,
            router: 5,
        }
    }

    #[test]
    fn full_and_all_sum_every_component() {
        let c = sample_components();
        let expected = c.embed + c.attn + c.ffn + c.lm_head + c.router;
        assert_eq!(estimate_preset_bytes("full", &c), Some(expected));
        assert_eq!(estimate_preset_bytes("all", &c), Some(expected));
    }

    #[test]
    fn client_is_embed_plus_attn_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("client", &c), Some(c.embed + c.attn));
    }

    #[test]
    fn attn_preset_is_attn_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("attn", &c), Some(c.attn));
        assert_eq!(estimate_preset_bytes("attention", &c), Some(c.attn));
    }

    #[test]
    fn embed_preset_is_embed_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("embed", &c), Some(c.embed));
        assert_eq!(estimate_preset_bytes("embed-server", &c), Some(c.embed));
    }

    #[test]
    fn server_preset_is_embed_plus_ffn() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("server", &c), Some(c.embed + c.ffn));
    }

    #[test]
    fn browse_preset_is_embed_only_approximation() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("browse", &c), Some(c.embed));
    }

    #[test]
    fn router_preset_is_router_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("router", &c), Some(c.router));
    }

    #[test]
    fn expert_server_preset_excludes_attention() {
        let c = sample_components();
        assert_eq!(
            estimate_preset_bytes("expert-server", &c),
            Some(c.embed + c.ffn + c.router)
        );
    }

    #[test]
    fn unknown_preset_has_no_estimate_rather_than_zero() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("bogus", &c), None);
    }

    #[test]
    fn preset_name_matching_is_case_insensitive() {
        let c = sample_components();
        assert_eq!(
            estimate_preset_bytes("FULL", &c),
            estimate_preset_bytes("full", &c)
        );
    }
}
