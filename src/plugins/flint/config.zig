//! Recursive Flint's claim and parse: the attention-MoE family's (`plugins/laguna/config.zig`) with Flint's reading
//! (NoPE from `no_rope_layer_indices`, sliding layers at `rope_theta`, renormed sigmoid routing). MLX-free.

const sdk = @import("../../sdk/root.zig");
const fam = @import("../laguna/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "flint",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"RecursiveSI/flint-int4"},
};

pub const model_types = [_][]const u8{"flint"};
pub const claims = sdk.claimsByType(&model_types);
pub const parse = fam.Parser(.flint).parse;
pub const freeParsed = fam.freeParsed;

// ── tests ──

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const testing = std.testing;

/// A trimmed copy of RecursiveSI/flint-int4's config.json.
pub const flint_int4 =
    \\{"model_type":"flint","hidden_size":2048,"intermediate_size":6144,"moe_intermediate_size":512,
    \\"shared_expert_intermediate_size":1024,"num_hidden_layers":6,"num_attention_heads":32,"num_key_value_heads":4,
    \\"head_dim":128,"num_experts":256,"num_experts_per_tok":6,"norm_topk_prob":true,"vocab_size":150144,
    \\"rms_norm_eps":1e-06,"rope_theta":10000.0,"sliding_window":513,"use_sliding_window":true,
    \\"layer_types":["full_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\"no_rope_layer_indices":[0,1,5],"mlp_only_layers":[0,1],"router_score_function":"sigmoid","softmax_type":"learnable",
    \\"eos_token_id":[150001,150003,150005,150018],"tie_word_embeddings":false,"quantization":{"group_size":64,"bits":4}}
;

test "flint config: NoPE layers, sliding layers at rope_theta, renormed routing, batched decode declared" {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = flint_int4 }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(flint_int4), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const c = fam.of(&p);
    try testing.expect(c.no_rope[0] and c.no_rope[5] and !c.no_rope[2]);
    try testing.expectEqual(@as(f32, 10000.0), c.rope_local_base_freq);
    try testing.expect(c.route_norm and c.route_scale == 1.0);
    try testing.expect(p.caps.batches_decode);
    try testing.expectEqual(@as(u32, 4), p.shell.eos_count);
}
