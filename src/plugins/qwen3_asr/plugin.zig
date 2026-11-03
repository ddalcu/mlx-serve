//! Qwen3-ASR (and its forced aligner): the qwen3 plugin's dense trunk with the audio tower's rows spliced at the
//! audio placeholders. The tower, the transcription flow and the aligner's timestamp head stay the engine's
//! (`audio_tower.zig`, `forced_align.zig`): they read the served trunk through the shell (hidden rows and the head
//! `trunk_weight` lends it).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const q3 = @import("../qwen3/plugin.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = blk: {
    var t = q3.table;
    t.name = abi.Str.of(plugin.name);
    t.claims = config.claims;
    t.parse = &config.parse;
    break :blk t;
};

/// The tiny pack the conformance suite runs: two layers, width 64, tied, audio id 60.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var fx = try q3.conformanceFixture(a);
    fx.files = try a.dupe(sdk.fixture.File, fx.files);
    for (@constCast(fx.files)) |*f| if (std.mem.eql(u8, f.name, "config.json")) {
        f.bytes = try std.mem.concat(a, u8, &.{
            "{\"model_type\":\"qwen3_asr\",\"thinker_config\":{\"model_type\":\"qwen3_asr\",\"audio_token_id\":60,",
            "\"audio_config\":{\"d_model\":64,\"encoder_attention_heads\":2,\"n_window\":2,\"n_window_infer\":4,\"downsample_hidden_size\":8},",
            "\"text_config\":",
            f.bytes,
            "}}",
        });
    };
    return fx;
}

// ── tests ──

test {
    _ = config;
}
