//! Dense walk-FFN wire contract shared by the client (`larql-inference`),
//! the server and the router. One definition so the three cannot drift.

/// Dense-FFN endpoint.
pub const PATH: &str = "/v1/walk-ffn";

/// Content type of the binary f32 frame.
pub const BINARY_CT: &str = "application/x-larql-ffn";

/// First u32 of a binary request/response body. When equal to this marker
/// the frame is a batch (`marker + count + …`); otherwise the first u32 is
/// the layer index of a single-layer frame.
pub const BATCH_MARKER: u32 = 0xFFFF_FFFF;
