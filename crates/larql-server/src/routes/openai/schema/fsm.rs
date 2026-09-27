//! Schema-typed JSON state machine.
//!
//! Walks a [`Schema`] one character at a time. The FSM mutates only on
//! accepted characters; on `Reject`, callers can discard the
//! simulation without rolling back. This is critical for the per-token
//! mask path, which forks the FSM thousands of times per generation
//! step.
//!
//! ## Branch semantics for `OneOf`
//!
//! `Schema::OneOf` is implemented by carrying a `Vec<Fsm>` of parallel
//! sub-FSMs in a single `Frame::OneOf`. On `step`, each sub-FSM is
//! forked and stepped; if zero branches survive, the parent rejects.
//! If one survives, the OneOf frame is replaced by that branch's
//! single-frame stack (commit). If multiple survive, the OneOf frame
//! is updated with the trimmed branches.
//!
//! ## Termination
//!
//! `is_complete()` returns true exactly once the root value has fully
//! parsed and only whitespace (or EOS) is acceptable. The mask path
//! uses this to gate EOS: while `!is_complete()`, EOS tokens are
//! masked out so the model can't truncate mid-structure.

use super::ast::{ArraySchema, NumberSchema, ObjectSchema, Schema, StringSchema};

mod atoms;
use atoms::*;

/// Public step result. The FSM is left mutated only on `Ok`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Ok,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keyword {
    True,
    False,
    Null,
}

impl Keyword {
    fn bytes(self) -> &'static [u8] {
        match self {
            Keyword::True => b"true",
            Keyword::False => b"false",
            Keyword::Null => b"null",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberPhase {
    /// Saw `-` or first digit; awaiting more digits, `.`, `e/E`, or
    /// terminator.
    IntPart,
    /// Saw `.`, expecting at least one fraction digit.
    FracStart,
    FracDigits,
    /// Saw `e`/`E`, expecting `+`/`-` or first exponent digit.
    ExpStart,
    /// Saw exponent sign; need at least one digit.
    ExpSign,
    ExpDigits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectPhase {
    /// Just opened — expecting either `}` (empty obj) or `"` (key).
    AfterOpen,
    /// Inside a key string — handled by a nested `String` frame.
    InKey,
    /// Saw closing key quote; expecting `:`.
    ExpectColon,
    /// Saw `:`; expecting a value.
    ExpectValue,
    /// Inside the value — handled by a nested frame.
    InValue,
    /// Saw `,` after a value — a key MUST follow (`}` here would be a
    /// trailing comma, which JSON forbids).
    ExpectKey,
    /// Saw value's closing structure; expecting `,` or `}`.
    AfterValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayPhase {
    /// Just opened — expecting `]` (empty) or first value.
    AfterOpen,
    /// Inside a value — handled by a nested frame.
    InValue,
    /// Saw value's close; expecting `,` or `]`.
    AfterValue,
}

#[derive(Debug, Clone)]
enum Frame {
    /// Awaiting the start of a value matching this schema. We're in this
    /// frame *before* dispatching on the first character. Once the first
    /// char arrives we either resolve (e.g. to a Number frame) or reject.
    Value(Schema),
    Object(ObjectFrame),
    Array(ArrayFrame),
    String(StringFrame),
    Number(NumberFrame),
    Keyword(KeywordFrame),
    Const(ConstFrame),
    OneOf(OneOfFrame),
}

#[derive(Debug, Clone)]
struct ObjectFrame {
    spec: ObjectSchema,
    phase: ObjectPhase,
    /// Names of keys we've consumed and whose values we've parsed.
    seen: Vec<String>,
    /// Currently-being-parsed key string buffer (when `phase == InKey`).
    key_buf: Option<String>,
    /// Schema for the active value (when `phase == InValue`); set on
    /// transition out of `ExpectValue`.
    active_value: Option<Box<Schema>>,
}

#[derive(Debug, Clone)]
struct ArrayFrame {
    spec: ArraySchema,
    phase: ArrayPhase,
    count: usize,
}

#[derive(Debug, Clone)]
struct StringFrame {
    spec: StringSchema,
    /// True if this string is being consumed as an object key — when it
    /// closes, we re-enter ObjectFrame::ExpectColon instead of completing
    /// a value.
    is_key: bool,
    /// Decoded characters consumed so far (after escape handling). Used
    /// for enum / const matching.
    decoded: String,
    in_escape: bool,
    /// Hex digits remaining in a `\uXXXX` escape (4 → 0).
    unicode_left: u8,
}

#[derive(Debug, Clone)]
struct NumberFrame {
    spec: NumberSchema,
    phase: NumberPhase,
    digits: String,
}

#[derive(Debug, Clone)]
struct KeywordFrame {
    target: Keyword,
    /// Index of the next char to match.
    index: u8,
}

#[derive(Debug, Clone)]
struct ConstFrame {
    /// JSON-stringified constant value (canonical form via serde_json).
    target: Vec<char>,
    index: usize,
}

#[derive(Debug, Clone)]
struct OneOfFrame {
    /// Each branch is its own sub-FSM at the value-start point.
    branches: Vec<Fsm>,
}

/// Top-level state machine.
#[derive(Debug, Clone)]
pub struct Fsm {
    stack: Vec<Frame>,
    /// True when the root value has fully closed.
    done: bool,
}

impl Fsm {
    /// Construct an FSM expecting a single value matching `schema`.
    pub fn new(schema: Schema) -> Self {
        Self {
            stack: vec![Frame::Value(schema)],
            done: false,
        }
    }

    /// FSM with `Schema::Any` — accepts any structurally-valid JSON.
    pub fn any() -> Self {
        Self::new(Schema::Any)
    }

    /// True iff the root value has been fully parsed and no further
    /// characters except whitespace are required.
    ///
    /// Numbers are special: they only naturally complete on a terminator
    /// (whitespace, `,`, `}`, `]`). A top-level bare number like `42`
    /// would otherwise sit forever in `IntPart` waiting for a delimiter.
    /// We treat a root-level Number frame in a valid end-phase (IntPart
    /// with ≥1 digit, FracDigits, ExpDigits) as complete-pending-EOS.
    pub fn is_complete(&self) -> bool {
        if self.done && self.stack.is_empty() {
            return true;
        }
        if self.stack.len() == 1 {
            if let Some(Frame::Number(n)) = self.stack.first() {
                return is_number_finalizable(n);
            }
        }
        false
    }

    /// Open container depth — `0` after the root closes. Used by
    /// callers that want a quick "is this still inside an object?" check.
    pub fn depth(&self) -> usize {
        self.stack
            .iter()
            .filter(|f| matches!(f, Frame::Object(_) | Frame::Array(_)))
            .count()
    }

    /// Apply one input character. The FSM mutates only on `Ok`.
    pub fn step(&mut self, ch: char) -> StepResult {
        // Whitespace handling: legal between top-level structural tokens
        // and around values inside containers. A few frames (String,
        // Number-active, Keyword-active, Const) consume whitespace as
        // part of their atom — those frames must short-circuit before
        // we hit this branch.
        if self.is_atomic_active() {
            return self.step_active_atom(ch);
        }
        if ch.is_ascii_whitespace() {
            // Whitespace is fine pre-root, between values, post-root.
            return StepResult::Ok;
        }
        // Pre-root: stack is exactly [Value(_)]. After root completes,
        // stack is empty and `done == true`.
        if self.done {
            return StepResult::Reject;
        }
        self.dispatch(ch)
    }

    /// Apply a sequence of characters. Stops at the first reject.
    pub fn step_str(&mut self, s: &str) -> StepResult {
        for ch in s.chars() {
            if self.step(ch) == StepResult::Reject {
                return StepResult::Reject;
            }
        }
        StepResult::Ok
    }

    fn is_atomic_active(&self) -> bool {
        matches!(
            self.stack.last(),
            Some(Frame::String(_) | Frame::Number(_) | Frame::Keyword(_) | Frame::Const(_))
        )
    }

    fn step_active_atom(&mut self, ch: char) -> StepResult {
        // Borrow the active frame mutably to advance, then check if
        // the atom completed.
        let last = self.stack.len() - 1;
        let outcome = match &mut self.stack[last] {
            Frame::String(s) => step_string(s, ch),
            Frame::Number(n) => step_number(n, ch),
            Frame::Keyword(k) => step_keyword(k, ch),
            Frame::Const(c) => step_const(c, ch),
            _ => unreachable!("is_atomic_active checked"),
        };
        match outcome {
            AtomOutcome::Ok => StepResult::Ok,
            AtomOutcome::Reject => StepResult::Reject,
            AtomOutcome::CompleteValue => {
                self.stack.pop();
                self.complete_value();
                StepResult::Ok
            }
            AtomOutcome::CompleteKey(key) => {
                // String was an object key; re-enter the parent
                // ObjectFrame::ExpectColon.
                self.stack.pop();
                if let Some(Frame::Object(obj)) = self.stack.last_mut() {
                    obj.phase = ObjectPhase::ExpectColon;
                    obj.key_buf = Some(key);
                    StepResult::Ok
                } else {
                    StepResult::Reject
                }
            }
            AtomOutcome::ReprocessAfterComplete(ch) => {
                // Number atoms consume terminators (`,`, `}`, etc.) by
                // first completing themselves and then asking the FSM
                // to re-handle the terminator at the parent level.
                self.stack.pop();
                self.complete_value();
                self.step(ch)
            }
        }
    }

    /// Top-level dispatch when the active frame isn't an atom.
    fn dispatch(&mut self, ch: char) -> StepResult {
        let Some(top) = self.stack.last() else {
            // Root completed — only whitespace allowed (handled above).
            return StepResult::Reject;
        };
        match top {
            Frame::Value(_) => self.dispatch_value(ch),
            Frame::Object(_) => self.dispatch_object(ch),
            Frame::Array(_) => self.dispatch_array(ch),
            Frame::OneOf(_) => self.dispatch_oneof(ch),
            // Atom frames are handled in step_active_atom.
            Frame::String(_) | Frame::Number(_) | Frame::Keyword(_) | Frame::Const(_) => {
                unreachable!("atom frames handled by step_active_atom")
            }
        }
    }

    fn dispatch_value(&mut self, ch: char) -> StepResult {
        let Some(Frame::Value(schema)) = self.stack.last() else {
            return StepResult::Reject;
        };
        let schema = schema.clone();
        // Replace the Value frame with the appropriate atom/container
        // frame, conditioned on `ch`.
        match (&schema, ch) {
            (Schema::Any, '{') => {
                self.replace_top(Frame::Object(ObjectFrame {
                    spec: ObjectSchema::any(),
                    phase: ObjectPhase::AfterOpen,
                    seen: Vec::new(),
                    key_buf: None,
                    active_value: None,
                }));
                StepResult::Ok
            }
            (Schema::Object(spec), '{') => {
                self.replace_top(Frame::Object(ObjectFrame {
                    spec: spec.clone(),
                    phase: ObjectPhase::AfterOpen,
                    seen: Vec::new(),
                    key_buf: None,
                    active_value: None,
                }));
                StepResult::Ok
            }
            (Schema::Any, '[') => {
                self.replace_top(Frame::Array(ArrayFrame {
                    spec: ArraySchema {
                        items: Box::new(Schema::Any),
                        min: None,
                        max: None,
                    },
                    phase: ArrayPhase::AfterOpen,
                    count: 0,
                }));
                StepResult::Ok
            }
            (Schema::Array(spec), '[') => {
                self.replace_top(Frame::Array(ArrayFrame {
                    spec: spec.clone(),
                    phase: ArrayPhase::AfterOpen,
                    count: 0,
                }));
                StepResult::Ok
            }
            (Schema::Any, '"') | (Schema::String(_), '"') => {
                let spec = match &schema {
                    Schema::String(s) => s.clone(),
                    _ => StringSchema::default(),
                };
                self.replace_top(Frame::String(StringFrame {
                    spec,
                    is_key: false,
                    decoded: String::new(),
                    in_escape: false,
                    unicode_left: 0,
                }));
                StepResult::Ok
            }
            (Schema::Any, c) | (Schema::Number(_), c) if c == '-' || c.is_ascii_digit() => {
                let spec = match &schema {
                    Schema::Number(n) => n.clone(),
                    _ => NumberSchema::default(),
                };
                let digits = String::from(c);
                self.replace_top(Frame::Number(NumberFrame {
                    spec,
                    phase: NumberPhase::IntPart,
                    digits,
                }));
                StepResult::Ok
            }
            (Schema::Any, 't') | (Schema::Boolean, 't') => {
                self.replace_top(Frame::Keyword(KeywordFrame {
                    target: Keyword::True,
                    index: 1,
                }));
                StepResult::Ok
            }
            (Schema::Any, 'f') | (Schema::Boolean, 'f') => {
                self.replace_top(Frame::Keyword(KeywordFrame {
                    target: Keyword::False,
                    index: 1,
                }));
                StepResult::Ok
            }
            (Schema::Any, 'n') | (Schema::Null, 'n') => {
                self.replace_top(Frame::Keyword(KeywordFrame {
                    target: Keyword::Null,
                    index: 1,
                }));
                StepResult::Ok
            }
            (Schema::Const(value), c) => {
                // Render the const canonically (compact serde_json) and
                // verify the first char matches.
                let target: Vec<char> = serde_json::to_string(value)
                    .unwrap_or_default()
                    .chars()
                    .collect();
                if target.is_empty() || target[0] != c {
                    return StepResult::Reject;
                }
                if target.len() == 1 {
                    self.stack.pop();
                    self.complete_value();
                } else {
                    self.replace_top(Frame::Const(ConstFrame { target, index: 1 }));
                }
                StepResult::Ok
            }
            (Schema::OneOf(branches), _) => {
                // Lazily expand the OneOf into a OneOfFrame and route
                // the char to it.
                let sub_fsms: Vec<Fsm> = branches.iter().map(|b| Fsm::new(b.clone())).collect();
                self.replace_top(Frame::OneOf(OneOfFrame { branches: sub_fsms }));
                self.dispatch_oneof(ch)
            }
            _ => StepResult::Reject,
        }
    }

    fn dispatch_object(&mut self, ch: char) -> StepResult {
        // Snapshot the immutable spec / phase fields we need for routing.
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return StepResult::Reject;
        };
        let phase = obj.phase;
        match phase {
            ObjectPhase::AfterOpen => match ch {
                '}' => self.close_object_if_required_satisfied(),
                '"' => {
                    // A closed object with no viable key left admits
                    // only `}` — refuse opening a key that could never
                    // close.
                    if self.viable_key_enum().is_some_and(|v| v.is_empty()) {
                        return StepResult::Reject;
                    }
                    self.set_object_phase(ObjectPhase::InKey);
                    self.push_string_frame_for_key();
                    StepResult::Ok
                }
                _ => StepResult::Reject,
            },
            ObjectPhase::ExpectColon => match ch {
                ':' => {
                    self.set_object_phase(ObjectPhase::ExpectValue);
                    StepResult::Ok
                }
                _ => StepResult::Reject,
            },
            ObjectPhase::ExpectValue => {
                // Resolve which schema applies to this key, push a Value
                // frame for it, then re-dispatch the char.
                let key = self.consume_object_key();
                let value_schema = match self.resolve_key_schema(&key) {
                    Ok(s) => s,
                    Err(()) => return StepResult::Reject,
                };
                self.set_object_phase(ObjectPhase::InValue);
                self.set_object_active_value(value_schema.clone());
                self.stack.push(Frame::Value(value_schema));
                // Re-dispatch the current char in the new value frame.
                self.dispatch_value(ch)
            }
            ObjectPhase::AfterValue => match ch {
                ',' => {
                    // A comma commits the emission to another key, so
                    // it is only legal while a key remains emittable —
                    // otherwise the mask would admit a `,` that can
                    // never be legally followed (found by the
                    // N0.6-on-V3 gate: greedy emitted `…{},}`).
                    if self.viable_key_enum().is_some_and(|v| v.is_empty()) {
                        return StepResult::Reject;
                    }
                    self.set_object_phase(ObjectPhase::ExpectKey);
                    StepResult::Ok
                }
                '}' => self.close_object_if_required_satisfied(),
                _ => StepResult::Reject,
            },
            ObjectPhase::ExpectKey => match ch {
                // Post-comma: a key is mandatory — `}` here would be a
                // trailing comma, which JSON forbids.
                '"' => {
                    self.set_object_phase(ObjectPhase::InKey);
                    self.push_string_frame_for_key();
                    StepResult::Ok
                }
                _ => StepResult::Reject,
            },
            ObjectPhase::InKey | ObjectPhase::InValue => {
                // The active frame should be the nested string/value;
                // routing was supposed to be handled by step_active_atom
                // or via the new top frame. We end up here only when
                // an Object frame's phase says "InValue" but the value
                // frame already popped — i.e., the value just completed
                // and we should be in AfterValue. Treat this as the
                // post-value path.
                StepResult::Reject
            }
        }
    }

    fn dispatch_array(&mut self, ch: char) -> StepResult {
        let Some(Frame::Array(arr)) = self.stack.last() else {
            return StepResult::Reject;
        };
        let phase = arr.phase;
        match phase {
            ArrayPhase::AfterOpen => match ch {
                ']' => self.close_array_if_within_bounds(),
                _ => {
                    // Any value char — push a Value frame typed by items.
                    let item = (*arr.spec.items).clone();
                    self.set_array_phase(ArrayPhase::InValue);
                    self.stack.push(Frame::Value(item));
                    self.dispatch_value(ch)
                }
            },
            ArrayPhase::AfterValue => match ch {
                ',' => {
                    let item = match self.stack.last() {
                        Some(Frame::Array(arr)) => (*arr.spec.items).clone(),
                        _ => return StepResult::Reject,
                    };
                    self.set_array_phase(ArrayPhase::InValue);
                    self.stack.push(Frame::Value(item));
                    StepResult::Ok
                }
                ']' => self.close_array_if_within_bounds(),
                _ => StepResult::Reject,
            },
            ArrayPhase::InValue => StepResult::Reject,
        }
    }

    fn dispatch_oneof(&mut self, ch: char) -> StepResult {
        let Some(Frame::OneOf(oo)) = self.stack.last() else {
            return StepResult::Reject;
        };
        // Step every branch; keep survivors.
        let mut surviving: Vec<Fsm> = Vec::new();
        for branch in &oo.branches {
            let mut probe = branch.clone();
            if probe.step(ch) == StepResult::Ok {
                surviving.push(probe);
            }
        }
        if surviving.is_empty() {
            return StepResult::Reject;
        }
        if surviving.len() == 1 {
            // Commit: pop the OneOf frame and splice the sub-FSM's
            // stack into ours. We can't use `sub.is_complete()` here
            // because that treats a root-level Number-in-progress as
            // complete (since EOS would be valid) — the model may still
            // want to extend the atom, so we keep the frame around.
            // Only propagate `done` when the sub-FSM has actually
            // emptied its stack (e.g. completed a keyword like `null`).
            let mut sub = surviving.into_iter().next().unwrap();
            self.stack.pop();
            let sub_done = sub.done;
            let sub_was_empty = sub.stack.is_empty();
            self.stack.append(&mut sub.stack);
            if sub_done && sub_was_empty {
                self.complete_value();
            }
            StepResult::Ok
        } else {
            // Multiple branches still alive — replace the OneOf frame
            // with the trimmed list.
            self.replace_top(Frame::OneOf(OneOfFrame {
                branches: surviving,
            }));
            StepResult::Ok
        }
    }

    fn replace_top(&mut self, frame: Frame) {
        if let Some(last) = self.stack.last_mut() {
            *last = frame;
        }
    }

    fn set_object_phase(&mut self, phase: ObjectPhase) {
        if let Some(Frame::Object(obj)) = self.stack.last_mut() {
            obj.phase = phase;
        }
    }

    fn set_object_active_value(&mut self, schema: Schema) {
        if let Some(Frame::Object(obj)) = self.stack.last_mut() {
            obj.active_value = Some(Box::new(schema));
        }
    }

    fn set_array_phase(&mut self, phase: ArrayPhase) {
        if let Some(Frame::Array(arr)) = self.stack.last_mut() {
            arr.phase = phase;
        }
    }

    fn push_string_frame_for_key(&mut self) {
        // On a CLOSED object (`additionalProperties: false`) the key
        // string is constrained to the still-viable property names via
        // the string frame's enum machinery — prefix-filtered while
        // the key is being emitted, exact-matched at the closing
        // quote. Without this the mask would admit a doomed key and
        // dead-end after its colon (found by the N0.6-on-V3 gate: the
        // constrained emission produced `{"name":"get","get":` and
        // starved). Open objects keep unconstrained keys.
        self.stack.push(Frame::String(StringFrame {
            spec: StringSchema {
                r#enum: self.viable_key_enum(),
                ..StringSchema::default()
            },
            is_key: true,
            decoded: String::new(),
            in_escape: false,
            unicode_left: 0,
        }));
    }

    /// Property names still emittable in the current Object frame, for
    /// constraining key strings at emission time. `None` means
    /// unconstrained (the object admits additional properties). Keys
    /// already seen are excluded — re-emitting one would be valid JSON
    /// but can never make progress toward closing the object.
    fn viable_key_enum(&self) -> Option<Vec<String>> {
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return None;
        };
        if obj.spec.additional.is_some() {
            return None;
        }
        Some(
            obj.spec
                .properties
                .keys()
                .filter(|k| !obj.seen.iter().any(|seen| &seen == k))
                .cloned()
                .collect(),
        )
    }

    fn consume_object_key(&mut self) -> String {
        if let Some(Frame::Object(obj)) = self.stack.last_mut() {
            let key = obj.key_buf.take().unwrap_or_default();
            obj.seen.push(key.clone());
            key
        } else {
            String::new()
        }
    }

    /// Look up the schema that applies to `key` in the current Object
    /// frame. Returns Err on unknown-key when `additionalProperties:
    /// false`.
    fn resolve_key_schema(&self, key: &str) -> Result<Schema, ()> {
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return Err(());
        };
        if let Some(schema) = obj.spec.properties.get(key) {
            return Ok(schema.clone());
        }
        match &obj.spec.additional {
            Some(s) => Ok((**s).clone()),
            None => Err(()),
        }
    }

    fn close_object_if_required_satisfied(&mut self) -> StepResult {
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return StepResult::Reject;
        };
        for req in &obj.spec.required {
            if !obj.seen.iter().any(|k| k == req) {
                return StepResult::Reject;
            }
        }
        self.stack.pop();
        self.complete_value();
        StepResult::Ok
    }

    fn close_array_if_within_bounds(&mut self) -> StepResult {
        let Some(Frame::Array(arr)) = self.stack.last() else {
            return StepResult::Reject;
        };
        if let Some(min) = arr.spec.min {
            if arr.count < min {
                return StepResult::Reject;
            }
        }
        self.stack.pop();
        self.complete_value();
        StepResult::Ok
    }

    /// Called after a value (string, number, keyword, container) has
    /// fully closed. Updates the parent frame to its post-value state.
    fn complete_value(&mut self) {
        // If the parent is an Object, we just finished the active value.
        // If parent is Array, increment count and move to AfterValue.
        // If no parent, root is done.
        if self.stack.is_empty() {
            self.done = true;
            return;
        }
        match self.stack.last_mut() {
            Some(Frame::Object(obj)) => {
                obj.phase = ObjectPhase::AfterValue;
                obj.active_value = None;
            }
            Some(Frame::Array(arr)) => {
                arr.count += 1;
                if let Some(max) = arr.spec.max {
                    if arr.count > max {
                        // Caller should have rejected before adding the
                        // value, but defend here: leave the FSM in a
                        // state where the next char can't be parsed.
                    }
                    let _ = max;
                }
                arr.phase = ArrayPhase::AfterValue;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
