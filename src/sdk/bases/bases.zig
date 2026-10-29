//! The language-model bases and the blocks they ship with. A plugin composes a base at comptime:
//! `bases.Decoder(.{ .embed = ..., .norm = ..., .mixer = ..., .mlp = ..., .seam = ..., .head = ... })`.

pub const block = @import("block.zig");
pub const kv = @import("kv.zig");
pub const run = @import("run.zig");
pub const Decoder = @import("decoder.zig").Decoder;
pub const Hybrid = @import("hybrid.zig").Hybrid;
pub const Encoder = @import("encoder.zig").Encoder;

pub const Ctx = block.Ctx;
pub const Shared = block.Shared;
pub const Either = block.Either;
pub const Model = run.Model;

pub const blocks = struct {
    pub const linear = @import("blocks/linear.zig");
    pub const embed = @import("blocks/embed.zig");
    pub const norm = @import("blocks/norm.zig");
    pub const attention = @import("blocks/attention.zig");
    pub const mlp = @import("blocks/mlp.zig");
    pub const seam = @import("blocks/seam.zig");
    pub const head = @import("blocks/head.zig");
    pub const gdn = @import("blocks/gdn.zig");
    pub const gated_attention = @import("blocks/gated_attention.zig");
    pub const moe = @import("blocks/moe.zig");
    pub const qk_rope = @import("blocks/qk_rope.zig");
    pub const conv_window = @import("blocks/conv_window.zig");
    pub const gated_conv = @import("blocks/gated_conv.zig");
    pub const mamba2 = @import("blocks/mamba2.zig");
    pub const nemotron_moe = @import("blocks/nemotron_moe.zig");
    pub const sliding = @import("blocks/sliding.zig");
    pub const decoder_attention = @import("blocks/decoder_attention.zig");
    pub const laguna_attention = @import("blocks/laguna_attention.zig");
    pub const extras = @import("blocks/extras.zig");
    pub const decode_copies = @import("blocks/decode_copies.zig");
    pub const trunk_weight = @import("blocks/trunk_weight.zig");
};

// ── tests ──

test {
    _ = block;
    _ = @import("../testing/cache_ops.zig");
}
