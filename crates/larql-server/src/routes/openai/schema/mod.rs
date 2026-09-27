//! Schema-typed JSON constrained decoding for the OpenAI routes.
//!
//! The AST, parser, FSM and logit mask live in
//! [`larql_inference::constrained`] so every front-end can decode against
//! a JSON Schema; they are re-exported here so the routes keep one import
//! path. [`tools`] is the OpenAI-specific part: tool-call schema synthesis
//! and `tool_choice` resolution.

pub mod tools;

pub use larql_inference::constrained::{
    ast, build_mask, fsm, mask, parse_schema, parse_schema_with, parser, ArraySchema, Fsm,
    NumberSchema, ObjectSchema, ParseOptions, Schema, StepResult, StringSchema,
};
pub use tools::{resolve_tool_choice, synth_tools_schema, ToolMode};
