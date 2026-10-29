//! The embedded llama.cpp engine's launch settings (`--llama-*`), read by the host's flags and the engine's load.

const std = @import("std");

/// Launch settings of the embedded llama.cpp engine (`--llama-*`); the adapter's `settings_json`.
pub const Settings = struct {
    /// Sequences per model: requests decoded together in one batch, each
    /// keeping its own prompt KV for prefix reuse.
    seqs: u32 = 4,
    kv_quant: LlamaKvQuant = .off,
    /// Physical prefill batch; 0 = libllama's default.
    ubatch: u32 = 0,
    /// Draft tokens per MTP round when the model ships a head; 0 = off.
    mtp_drafts: u32 = 2,
};

/// KV quant for the llama.cpp engine. Mapped from the CLI flag onto ggml types;
/// F16 is the dense default, Q8_0 halves KV bytes, Q4_0 quarters them.
pub const LlamaKvQuant = enum(u8) {
    off, // F16 (default)
    q8, // Q8_0 (~2x compression, near-lossless on most archs)
    q4, // Q4_0 (~4x compression, some quality impact)

    pub fn fromString(s: []const u8) ?LlamaKvQuant {
        if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "f16") or std.mem.eql(u8, s, "F16")) return .off;
        if (std.mem.eql(u8, s, "8") or std.mem.eql(u8, s, "q8") or std.mem.eql(u8, s, "Q8_0") or std.mem.eql(u8, s, "q8_0")) return .q8;
        if (std.mem.eql(u8, s, "4") or std.mem.eql(u8, s, "q4") or std.mem.eql(u8, s, "Q4_0") or std.mem.eql(u8, s, "q4_0")) return .q4;
        return null;
    }

    pub fn ggmlType(self: LlamaKvQuant) i32 {
        return switch (self) {
            .off => 0, // shim treats 0 as "use libllama default"
            .q8 => 8, // GGML_TYPE_Q8_0
            .q4 => 2, // GGML_TYPE_Q4_0
        };
    }

    pub fn label(self: LlamaKvQuant) []const u8 {
        return switch (self) {
            .off => "F16",
            .q8 => "Q8_0",
            .q4 => "Q4_0",
        };
    }
};
