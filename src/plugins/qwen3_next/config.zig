//! Qwen3-Next packs (`qwen3_next`, mlx-lm's layout): the claim, and the parse into `Parsed` through the qwen3_5
//! family's parser (GatedDeltaNet + gated attention + routed MLP with a gated shared expert). MLX-free.

const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const q35 = @import("../qwen3_5/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "qwen3_next",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"},
};

pub const model_types = [_][]const u8{"qwen3_next"};

pub const claims = sdk.claimsByType(&model_types);

pub const freeParsed = q35.freeParsed;
pub const layout = q35.layout;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return q35.parseAs(.next, dir, config_json, span, stub, out);
}

// ── tests ──

const std = @import("std");
const testing = std.testing;

test "qwen3_next config: Qwen3-Next-80B-A3B parses flat, serial, MoE, its rotary factor at the root" {
    const json =
        \\{"model_type":"qwen3_next","hidden_size":2048,"num_hidden_layers":48,"num_attention_heads":16,
        \\ "num_key_value_heads":2,"head_dim":256,"vocab_size":151936,"rms_norm_eps":1e-6,"rope_theta":10000000,
        \\ "partial_rotary_factor":0.25,"full_attention_interval":4,"linear_num_key_heads":16,"linear_num_value_heads":32,
        \\ "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_experts":512,
        \\ "num_experts_per_tok":10,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,
        \\ "intermediate_size":5120,"max_position_embeddings":262144,"hidden_act":"silu","decoder_sparse_step":1,
        \\ "mlp_only_layers":[],"eos_token_id":151645,"quantization":{"group_size":64,"bits":4}}
    ;
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(!p.caps.batches_decode and p.caps.moe);
    try testing.expectEqualStrings("qwen3_next", p.shell.model_type.slice());
    try testing.expectEqual(@as(u32, 64), q35.of(&p).ropeDims());
    try testing.expect(q35.of(&p).next);
}
