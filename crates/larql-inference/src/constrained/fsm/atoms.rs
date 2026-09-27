//! Atom step helpers: strings, numbers, keywords and constants, one char at a time.

#[allow(unused_imports)]
use super::*;

// ── Atom step helpers ────────────────────────────────────────────────────────
//
// Each atom (string, number, keyword, const) advances independently of
// the parent frame; the result tells the caller whether the atom is done
// and how to drive the parent.

pub(super) enum AtomOutcome {
    Ok,
    Reject,
    /// The atom completed and the parent should treat it as a finished
    /// value. (Used for non-string atoms and for value-context strings.)
    CompleteValue,
    /// The atom was a string in key context; pass the decoded key up.
    CompleteKey(String),
    /// The atom completed mid-step on the previous char; the current
    /// char must be re-processed by the parent.
    ReprocessAfterComplete(char),
}

pub(super) fn step_string(s: &mut StringFrame, ch: char) -> AtomOutcome {
    if s.unicode_left > 0 {
        if ch.is_ascii_hexdigit() {
            s.unicode_left -= 1;
            // We don't actually decode the codepoint here — for
            // enum/const matching we'd need the literal char, but the
            // common cases (tool names etc.) don't involve unicode
            // escapes. Push a placeholder so length matching stays
            // sensible.
            s.decoded.push('\u{FFFD}');
            return AtomOutcome::Ok;
        }
        return AtomOutcome::Reject;
    }
    if s.in_escape {
        s.in_escape = false;
        let decoded = match ch {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{0008}',
            'f' => '\u{000C}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => {
                s.unicode_left = 4;
                return AtomOutcome::Ok;
            }
            _ => return AtomOutcome::Reject,
        };
        s.decoded.push(decoded);
        return ok_if_within_string_constraints(s);
    }
    match ch {
        '\\' => {
            s.in_escape = true;
            AtomOutcome::Ok
        }
        '"' => {
            // String closed — validate against enum / const / minLen.
            if let Some(c) = s.spec.r#const.as_ref() {
                if &s.decoded != c {
                    return AtomOutcome::Reject;
                }
            }
            if let Some(en) = s.spec.r#enum.as_ref() {
                if !en.iter().any(|v| v == &s.decoded) {
                    return AtomOutcome::Reject;
                }
            }
            if let Some(min) = s.spec.min_len {
                if s.decoded.chars().count() < min {
                    return AtomOutcome::Reject;
                }
            }
            if s.is_key {
                AtomOutcome::CompleteKey(std::mem::take(&mut s.decoded))
            } else {
                AtomOutcome::CompleteValue
            }
        }
        c if (c as u32) < 0x20 => AtomOutcome::Reject,
        c => {
            s.decoded.push(c);
            ok_if_within_string_constraints(s)
        }
    }
}

/// While the string is still open, check that adding this char hasn't
/// already broken the const / enum prefix expectation. This lets the
/// FSM reject invalid characters early during tool-name matching.
pub(super) fn ok_if_within_string_constraints(s: &StringFrame) -> AtomOutcome {
    if let Some(c) = s.spec.r#const.as_ref() {
        if !c.starts_with(&s.decoded) {
            return AtomOutcome::Reject;
        }
    }
    if let Some(en) = s.spec.r#enum.as_ref() {
        if !en.iter().any(|v| v.starts_with(&s.decoded)) {
            return AtomOutcome::Reject;
        }
    }
    if let Some(max) = s.spec.max_len {
        if s.decoded.chars().count() > max {
            return AtomOutcome::Reject;
        }
    }
    AtomOutcome::Ok
}

pub(super) fn step_number(n: &mut NumberFrame, ch: char) -> AtomOutcome {
    let terminator = ch.is_ascii_whitespace() || matches!(ch, ',' | '}' | ']' | ':');
    match n.phase {
        NumberPhase::IntPart => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                AtomOutcome::Ok
            }
            '.' => {
                if n.spec.integer {
                    return AtomOutcome::Reject;
                }
                n.digits.push(ch);
                n.phase = NumberPhase::FracStart;
                AtomOutcome::Ok
            }
            'e' | 'E' => {
                if n.spec.integer {
                    return AtomOutcome::Reject;
                }
                n.digits.push(ch);
                n.phase = NumberPhase::ExpStart;
                AtomOutcome::Ok
            }
            _ if terminator => {
                if !validate_number(n) {
                    return AtomOutcome::Reject;
                }
                AtomOutcome::ReprocessAfterComplete(ch)
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::FracStart => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                n.phase = NumberPhase::FracDigits;
                AtomOutcome::Ok
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::FracDigits => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                AtomOutcome::Ok
            }
            'e' | 'E' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpStart;
                AtomOutcome::Ok
            }
            _ if terminator => {
                if !validate_number(n) {
                    return AtomOutcome::Reject;
                }
                AtomOutcome::ReprocessAfterComplete(ch)
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::ExpStart => match ch {
            '+' | '-' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpSign;
                AtomOutcome::Ok
            }
            '0'..='9' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpDigits;
                AtomOutcome::Ok
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::ExpSign => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpDigits;
                AtomOutcome::Ok
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::ExpDigits => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                AtomOutcome::Ok
            }
            _ if terminator => {
                if !validate_number(n) {
                    return AtomOutcome::Reject;
                }
                AtomOutcome::ReprocessAfterComplete(ch)
            }
            _ => AtomOutcome::Reject,
        },
    }
}

pub(super) fn validate_number(n: &NumberFrame) -> bool {
    let parsed: f64 = match n.digits.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if let Some(min) = n.spec.minimum {
        if parsed < min {
            return false;
        }
    }
    if let Some(max) = n.spec.maximum {
        if parsed > max {
            return false;
        }
    }
    true
}

/// True if the number atom is in a phase that could legally end here
/// (i.e. has at least one digit and isn't waiting on more required
/// characters like a fraction-digit or exponent-digit).
pub(super) fn is_number_finalizable(n: &NumberFrame) -> bool {
    let has_digit = n.digits.chars().any(|c| c.is_ascii_digit());
    if !has_digit {
        return false;
    }
    matches!(
        n.phase,
        NumberPhase::IntPart | NumberPhase::FracDigits | NumberPhase::ExpDigits
    ) && validate_number(n)
}

pub(super) fn step_keyword(k: &mut KeywordFrame, ch: char) -> AtomOutcome {
    let bytes = k.target.bytes();
    let idx = k.index as usize;
    if idx < bytes.len() && bytes[idx] as char == ch {
        let next = k.index + 1;
        if next as usize == bytes.len() {
            AtomOutcome::CompleteValue
        } else {
            k.index = next;
            AtomOutcome::Ok
        }
    } else {
        AtomOutcome::Reject
    }
}

pub(super) fn step_const(c: &mut ConstFrame, ch: char) -> AtomOutcome {
    if c.index >= c.target.len() {
        return AtomOutcome::Reject;
    }
    if c.target[c.index] != ch {
        return AtomOutcome::Reject;
    }
    c.index += 1;
    if c.index == c.target.len() {
        AtomOutcome::CompleteValue
    } else {
        AtomOutcome::Ok
    }
}
