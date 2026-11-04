//! Prism Bonsai packs (`prism_hadamard_qwen35`): the claim, and the parse through the qwen3_5 family's parser (a dense
//! Qwen3.5 trunk whose projections are stored in a Hadamard-rotated basis, `modules[].block`). MLX-free.

const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const q35 = @import("../qwen3_5/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "prism_hadamard_qwen35",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"},
};

pub const model_types = [_][]const u8{"prism_hadamard_qwen35"};

pub const claims = sdk.claimsByType(&model_types);

pub const freeParsed = q35.freeParsed;
pub const layout = q35.layout;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return q35.parseAs(.prism, dir, config_json, span, stub, out);
}

// ── tests ──

const std = @import("std");
const testing = std.testing;

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

const bonsai =
    \\{"model_type":"prism_hadamard_qwen35","tie_word_embeddings":false,"text_config":{"model_type":"qwen3_5_text",
    \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"num_attention_heads":24,
    \\ "num_key_value_heads":4,"head_dim":256,"vocab_size":248320,"rms_norm_eps":1e-6,"full_attention_interval":4,
    \\ "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
    \\ "linear_conv_kernel_dim":4,"hidden_act":"silu","eos_token_id":248044,
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}},
    \\ "modules":[{"path":"lm_head","block":1024},{"path":"model.layers.0.mlp.up_proj","block":1024}],
    \\ "quantization":{"group_size":128,"bits":2,"mode":"affine"}}
;

test "prism config: a Bonsai pack parses as the dense trunk in f16 with an f32 GDN state and MTP depth 2" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(bonsai, &p).isOk());
    defer freeParsed(&p);
    const c = q35.of(&p);
    try testing.expectEqual(@as(u32, 1024), c.hadamard_block);
    try testing.expect(p.caps.batches_decode and !p.caps.moe and !p.caps.exact_drafted and p.caps.dflash_capture);
    try testing.expect(p.shell.act_dtype == .f16 and p.shell.state_dtype == .f32 and p.cache.ssm_state.dtype == .f32);
    try testing.expectEqual(@as(u8, 2), p.spec.mtp_default_depth);
    try testing.expectEqualStrings("qwen3_5_moe", p.shell.model_type.slice());
    // 48 linear layers x (48 x 128 x 128 f32 state + 3 x 10240 bf16 conv rows).
    try testing.expectEqual(@as(u64, 48 * (48 * 128 * 128 * 4 + 3 * 10240 * 2)), p.cache.ssm_state.checkpoint_bytes);
}

test "prism config: modules without one shared block, and a qwen3_5 pack, are not this plugin's" {
    const mixed = try std.mem.replaceOwned(u8, testing.allocator, bonsai, "\"block\":1024}]", "\"block\":512}]");
    defer testing.allocator.free(mixed);
    const plain = try std.mem.replaceOwned(u8, testing.allocator, bonsai, "\"prism_hadamard_qwen35\"", "\"qwen3_5\"");
    defer testing.allocator.free(plain);
    for ([_][]const u8{ mixed, plain }) |json| {
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}
