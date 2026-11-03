//! Aleph Alpha Kolibri-1's claim and parse: the attention-MoE family's (`plugins/laguna/config.zig`) with Kolibri's
//! reading (NoPE on every full layer, sliding layers at `rope_theta`, top-k on biased logits, no renorm). MLX-free.

const sdk = @import("../../sdk/root.zig");
const fam = @import("../laguna/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "kolibri1",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const model_types = [_][]const u8{"kolibri1"};
pub const claims = sdk.claimsByType(&model_types);
pub const parse = fam.Parser(.kolibri1).parse;
pub const freeParsed = fam.freeParsed;

// ── tests ──

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const testing = std.testing;

/// A trimmed copy of Kolibri-1's config.json.
const kolibri =
    \\{"model_type":"kolibri1","hidden_size":2560,"num_hidden_layers":4,"num_attention_heads":48,"num_key_value_heads":4,
    \\"head_dim":128,"rms_norm_eps":1e-06,"vocab_size":128000,"max_position_embeddings":262144,"rope_theta":10000.0,
    \\"tie_word_embeddings":false,"use_sliding_window":true,"sliding_window":513,"num_experts":384,
    \\"num_experts_per_tok":6,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"norm_topk_prob":false,
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\"quantization":{"group_size":64,"bits":3,"mode":"affine"}}
;

test "kolibri1 config: NoPE full layers, sliding layers at rope_theta, unnormalized routing, serial" {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = kolibri }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(kolibri), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const c = fam.of(&p);
    try testing.expect(c.no_rope[3] and !c.no_rope[0]);
    try testing.expectEqual(@as(f32, 10000.0), c.rope_local_base_freq);
    try testing.expect(!c.route_norm);
    try testing.expect(!p.caps.batches_decode);
    try testing.expectEqual(@as(u8, 3), p.shell.quant.bits);
}
