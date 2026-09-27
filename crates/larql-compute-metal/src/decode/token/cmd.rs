//! The token's live command buffer and compute encoder.
//!
//! One `TokenCmd` exists per decode step. Stages that need to flush (a
//! staging blit, a profiling split, a diagnostic read) end the encoder,
//! commit, and reopen through here; `encoder_ended` records whether the
//! current encoder has already been ended so nothing ends it twice.

use metal::{CommandBuffer, CommandQueue, ComputeCommandEncoder};

pub(super) struct TokenCmd {
    // Field order is drop order: the encoder is released before its
    // command buffer, as the two locals this replaces were.
    pub enc: ComputeCommandEncoder,
    pub cmd: CommandBuffer,
    pub encoder_ended: bool,
}

impl TokenCmd {
    /// A fresh command buffer with one open compute encoder.
    pub(super) fn open(queue: &CommandQueue) -> Self {
        let cmd = queue.new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder().to_owned();
        Self {
            cmd,
            enc,
            encoder_ended: false,
        }
    }

    /// Replace the (already committed) command buffer with a fresh one and
    /// open a compute encoder on it.
    pub(super) fn reopen(&mut self, queue: &CommandQueue) {
        self.cmd = queue.new_command_buffer().to_owned();
        self.enc = self.cmd.new_compute_command_encoder().to_owned();
        self.encoder_ended = false;
    }

    /// Open a new compute encoder on the SAME command buffer, after a blit
    /// encoder has been interleaved into it.
    pub(super) fn reopen_encoder(&mut self) {
        self.enc = self.cmd.new_compute_command_encoder().to_owned();
        self.encoder_ended = false;
    }

    /// End the compute encoder unless something already did.
    pub(super) fn end_encoder_if_open(&self) {
        if !self.encoder_ended {
            self.enc.end_encoding();
        }
    }

    /// End the encoder, commit, and wait (refusing on failure). Leaves
    /// `encoder_ended` for the caller to set, as each flush site differs
    /// on whether it reopens.
    pub(super) fn commit_and_wait(&self, site: &'static str) {
        self.enc.end_encoding();
        self.cmd.commit();
        crate::cb_status::wait_or_abort(&self.cmd, site);
    }
}
