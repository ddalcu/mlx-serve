//! Prism Bonsai packs (`prism_hadamard_qwen35`): the claim, and the parse through the qwen3_5 family's parser (a dense
//! Qwen3.5 trunk whose projections are stored in a Hadamard-rotated basis, `modules[].block`). MLX-free.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const q3 = @import("../qwen3/config.zig");
const q35 = @import("../qwen3_5/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "prism_hadamard_qwen35",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"},
};

pub const model_types = [_][]const u8{"prism_hadamard_qwen35"};

/// Official packs by type. A `qwen3_5` pack only when it is affine 2-bit group 128 and names one rotation block,
/// which outranks qwen3_5's claim by type. A 2-bit pack without that block stays with qwen3_5 and is declined.
pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const mt = model_type.slice();
    if (std.mem.eql(u8, mt, "prism_hadamard_qwen35")) return .by_type;
    if (!std.mem.eql(u8, mt, "qwen3_5")) return .no;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const d = q3.doc(arena.allocator(), config_json.slice()) catch return .no;
    const q = q3.field(d.root, "quantization") orelse return .no;
    if (q != .object) return .no;
    const bits = q3.int(q3.field(q.object, "bits")) orelse return .no;
    const gs = q3.int(q3.field(q.object, "group_size")) orelse 64;
    if (bits != 2 or gs != 128) return .no;
    if (q3.field(q.object, "mode")) |m| {
        if (m != .string or !std.mem.eql(u8, m.string, "affine")) return .no;
    }
    if (q35.rotationBlock(d.root, dir) == null) return .no;
    return .by_marker;
}

pub const freeParsed = q35.freeParsed;
pub const layout = q35.layout;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return q35.parseAs(.prism, dir, config_json, span, stub, out);
}

// ── tests ──

const testing = std.testing;

fn parseFiles(files: []const sdk.fixture.File, json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    return parseFiles(&.{.{ .name = "config.json", .bytes = json }}, json, out);
}

fn claimFiles(files: []const sdk.fixture.File, model_type: []const u8, json: []const u8) abi.Claim {
    const mem: sdk.fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return claims(.of(model_type), .of(json), &dir);
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

fn expectPrism(json: []const u8, files: []const sdk.fixture.File) !void {
    var p: abi.Parsed = .{};
    const v = if (files.len == 0) parseJson(json, &p) else parseFiles(files, json, &p);
    try testing.expect(v.isOk());
    defer freeParsed(&p);
    const c = q35.of(&p);
    try testing.expectEqual(@as(u32, 1024), c.hadamard_block);
    try testing.expect(p.shell.act_dtype == .f16 and p.shell.state_dtype == .f32);
    try testing.expectEqual(@as(u8, 2), p.spec.mtp_default_depth);
}

test "prism config: modules without one shared block are not this plugin's; a qwen3_5 pack that names one is" {
    const mixed = try std.mem.replaceOwned(u8, testing.allocator, bonsai, "\"block\":1024}]", "\"block\":512}]");
    defer testing.allocator.free(mixed);
    var bad: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(mixed, &bad).code);

    const plain = try std.mem.replaceOwned(u8, testing.allocator, bonsai, "\"prism_hadamard_qwen35\"", "\"qwen3_5\"");
    defer testing.allocator.free(plain);
    try expectPrism(plain, &.{});
    try testing.expectEqual(abi.Claim.by_marker, claimFiles(&.{.{ .name = "config.json", .bytes = plain }}, "qwen3_5", plain));
    var ignored: abi.PackDir = undefined;
    try testing.expectEqual(abi.Claim.by_type, q35.claims(.of("qwen3_5"), .of(plain), &ignored));
}

test "prism config: a qwen3_5 pack claims by marker from hadamard.block_size or hadamard.json" {
    const renamed = try std.mem.replaceOwned(u8, testing.allocator, bonsai, "\"prism_hadamard_qwen35\"", "\"qwen3_5\"");
    defer testing.allocator.free(renamed);
    const stripped = try std.mem.replaceOwned(u8, testing.allocator, renamed, "\"modules\":[{\"path\":\"lm_head\",\"block\":1024},{\"path\":\"model.layers.0.mlp.up_proj\",\"block\":1024}],", "");
    defer testing.allocator.free(stripped);
    const object = try std.mem.replaceOwned(u8, testing.allocator, stripped, "\"quantization\"", "\"hadamard\":{\"block_size\":1024},\"quantization\"");
    defer testing.allocator.free(object);
    try expectPrism(object, &.{});
    try testing.expectEqual(abi.Claim.by_marker, claimFiles(&.{.{ .name = "config.json", .bytes = object }}, "qwen3_5", object));
    var ignored: abi.PackDir = undefined;
    try testing.expectEqual(abi.Claim.by_type, claims(.of("prism_hadamard_qwen35"), .of(bonsai), &ignored));

    const side = "{\"prism.hadamard.block_size\":1024}";
    const files = [_]sdk.fixture.File{
        .{ .name = "config.json", .bytes = stripped },
        .{ .name = "hadamard.json", .bytes = side },
    };
    try expectPrism(stripped, &files);
    try testing.expectEqual(abi.Claim.by_marker, claimFiles(&files, "qwen3_5", stripped));

    var late = try testing.allocator.alloc(u8, 4200);
    defer testing.allocator.free(late);
    @memset(late, ' ');
    const key = "\"prism.hadamard.block_size\":1024";
    @memcpy(late[4100..][0..key.len], key);
    const late_files = [_]sdk.fixture.File{
        .{ .name = "config.json", .bytes = stripped },
        .{ .name = "hadamard.json", .bytes = late },
    };
    var missed: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseFiles(&late_files, stripped, &missed).code);
    try testing.expectEqual(abi.Claim.no, claimFiles(&late_files, "qwen3_5", stripped));
}

test "prism config: a 2-bit qwen3_5 pack with no rotation block, and a 4-bit pack with one, are not claimed" {
    const renamed = try std.mem.replaceOwned(u8, testing.allocator, bonsai, "\"prism_hadamard_qwen35\"", "\"qwen3_5\"");
    defer testing.allocator.free(renamed);
    const bare = try std.mem.replaceOwned(u8, testing.allocator, renamed, "\"modules\":[{\"path\":\"lm_head\",\"block\":1024},{\"path\":\"model.layers.0.mlp.up_proj\",\"block\":1024}],", "");
    defer testing.allocator.free(bare);
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(bare, &p).code);
    try testing.expectEqual(abi.Claim.no, claimFiles(&.{.{ .name = "config.json", .bytes = bare }}, "qwen3_5", bare));

    const wide = try std.mem.replaceOwned(u8, testing.allocator, bare, "\"bits\":2", "\"bits\":4");
    defer testing.allocator.free(wide);
    const marked = try std.mem.replaceOwned(u8, testing.allocator, wide, "\"quantization\"", "\"hadamard\":{\"block_size\":1024},\"quantization\"");
    defer testing.allocator.free(marked);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(marked, &p).code);
    try testing.expectEqual(abi.Claim.no, claimFiles(&.{.{ .name = "config.json", .bytes = marked }}, "qwen3_5", marked));

    const expand =
        \\{"requires_jang_affine1_expansion": true}
    ;
    const object = try std.mem.replaceOwned(u8, testing.allocator, bare, "\"quantization\"", "\"hadamard\":{\"block_size\":1024},\"quantization\"");
    defer testing.allocator.free(object);
    const files = [_]sdk.fixture.File{
        .{ .name = "config.json", .bytes = object },
        .{ .name = "jang_config.json", .bytes = expand },
    };
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseFiles(&files, object, &p).code);
}
