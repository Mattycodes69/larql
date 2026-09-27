//! Schema-typed JSON constrained decoding.
//!
//! The pipeline:
//!
//! 1. **AST** ([`ast`]) — typed Schema enum (`Object`, `Array`, `String`,
//!    `Number`, `OneOf`, `Const`, etc.) the FSM walks.
//! 2. **Parser** ([`parser`]) — JSON Schema → AST.
//! 3. **FSM** ([`fsm`]) — character-level state machine that consumes
//!    JSON and rejects anything that diverges from the schema.
//! 4. **Mask** ([`mask`]) — adapter that wraps the FSM into the
//!    `FnMut(&[u32], &mut Vec<f32>)` signature
//!    [`crate::layer_graph::generate_constrained`] expects.
//!
//! Front-ends build on this: the server's OpenAI routes add tool-call
//! schema synthesis on top of the AST.

pub mod ast;
pub mod fsm;
pub mod mask;
pub mod parser;

pub use ast::{ArraySchema, NumberSchema, ObjectSchema, Schema, StringSchema};
pub use fsm::{Fsm, StepResult};
pub use mask::build_mask;
pub use parser::{parse_schema, parse_schema_with, ParseOptions};
