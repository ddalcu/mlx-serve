//! `block.zig`'s tests, out of its import graph: a plugin unit that reaches `block.zig` never compiles them.

const block_mod = @import("block.zig");
const std = @import("std");
const abi = @import("../../abi/quant.zig");
const Verdict = abi.Verdict;
const blocksOf = block_mod.blocksOf;
const builtins = block_mod.builtins;
const claimant = block_mod.claimant;
const declared = block_mod.declared;
const hadamardBlock = block_mod.hadamardBlock;
const registry = @import("../../plugins/registry.zig");
const testing = std.testing;

fn expectClaimant(want: ?[]const u8, block_json: []const u8, config_json: []const u8) !void {
    const got = try claimant(block_json, config_json);
    const name: ?[]const u8 = if (got) |i| builtins[i].name else null;
    if (want) |w| {
        try testing.expect(name != null);
        try testing.expectEqualStrings(w, name.?);
    } else try testing.expectEqual(@as(?[]const u8, null), name);
}

// Bar: every block of every fixture config has at most one claimant, and the ones today's parser serves have theirs.
test "quant block: each fixture config's quantization block is claimed by one built-in format at most" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Blocks today's parser ignores (bitsandbytes, compressed-tensors, `modelopt_fp4`) have no claimant.
    const served = [_]struct { file: []const u8, key: []const u8, format: []const u8 }{
        .{ .file = "nemotron_h_super_omni.json", .key = "quantization", .format = "affine" },
        .{ .file = "nemotron_h_super_omni.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "glm5_next-mlx-community-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "glm5_next-mlx-community-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "glm5_next-nvidia-fp4-local.json", .key = "quantization_config", .format = "nvfp4" },
        .{ .file = "glm5_next-sushi-2.4bpw.json", .key = "quantization", .format = "affine" },
        .{ .file = "glm5_next-sushi-2.4bpw.json", .key = "expert_quant", .format = "exl3" },
        .{ .file = "bailing_hybrid-ling-3.0-flash-oq4e-mtp.json", .key = "quantization", .format = "affine" },
        .{ .file = "bailing_hybrid-ling-3.0-flash-oq4e-mtp.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "bailing_hybrid-ling-3.0-tiny-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "bailing_hybrid-ling-3.0-tiny-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "qwen4_exp-iq-mlx-4.7bpw.json", .key = "quantization", .format = "affine" },
        .{ .file = "qwen4_exp-iq-mlx-4.7bpw.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "qwen4_exp-nvidia-fp4.json", .key = "quantization_config", .format = "nvfp4" },
        .{ .file = "embeddinggemma-300m-8bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-3-12b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-3-1b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-3-1b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-3-4b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-12b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-12b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-4-26b-a4b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-26b-a4b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-4-31b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-31b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-4-e4b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-e4b-it-4bit.json", .key = "quantization_config", .format = "affine" },
    };
    var seen: usize = 0;
    for (registry.model_configs) |f| {
        for (try blocksOf(a, f.bytes)) |b| {
            const got = claimant(b.json, f.bytes) catch {
                std.debug.print("{s}: block {s} has two claimants\n", .{ f.name, b.key });
                return error.TwoClaimants;
            };
            var want: ?[]const u8 = null;
            for (served) |s| if (std.mem.eql(u8, s.file, f.name) and std.mem.eql(u8, s.key, b.key)) {
                want = s.format;
                seen += 1;
            };
            const name: ?[]const u8 = if (got) |i| builtins[i].name else null;
            if ((want == null) != (name == null) or (want != null and !std.mem.eql(u8, want.?, name.?))) {
                std.debug.print("{s}: block {s} claimed by {?s}, want {?s}\n", .{ f.name, b.key, name, want });
                return error.WrongClaimant;
            }
        }
    }
    try testing.expectEqual(served.len, seen);
}

// Bar: each shipped pack shape names exactly its format; Prism's `modules` outranks the affine mode beside it.
test "quant block: every shipped block shape has one claimant" {
    const none = "{}";
    try expectClaimant("affine", "{\"group_size\":64,\"bits\":4}", none);
    try expectClaimant("affine", "{\"group_size\":64,\"bits\":3,\"mode\":\"affine\",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8}}", none);
    try expectClaimant("mxfp8", "{\"group_size\":32,\"bits\":8,\"mode\":\"mxfp8\"}", none);
    try expectClaimant("nvfp4", "{\"group_size\":16,\"bits\":4,\"mode\":\"nvfp4\"}", none);
    try expectClaimant("nvfp4", "{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\",\"model.embed_tokens\":{\"group_size\":32,\"bits\":8,\"mode\":\"mxfp8\"}}", none);
    try expectClaimant("nvfp4", "{\"quant_method\":\"modelopt\",\"quant_algo\":\"MIXED_PRECISION\"}", none);
    try expectClaimant("fp8", "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", none);
    const prism = "{\"model_type\":\"prism_hadamard_qwen35\",\"modules\":[{\"path\":\"lm_head\",\"block\":1024}]}";
    try expectClaimant("hadamard", "{\"bits\":2,\"group_size\":128,\"mode\":\"affine\"}", prism);
    try expectClaimant("exl3", "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}", none);
    try expectClaimant("exl3", "{\"format\":\"stq\",\"k\":2.5,\"codebook\":\"std1\",\"window\":15,\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"}", none);
    try expectClaimant("gguf", "{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}", none);
    try expectClaimant(null, "{\"quant_method\":\"bitsandbytes\",\"load_in_4bit\":true}", none);
    try expectClaimant(null, "{\"group_size\":64}", none);
    try expectClaimant(null, "{", none);
}

test "quant block: blocks are found under their keys; a config without one has none" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bs = try blocksOf(a, "{\"expert_quant\":{\"format\":\"exl3\"},\"quantization\":{\"bits\":4},\"quantization_config\":3}");
    try testing.expectEqual(@as(usize, 2), bs.len);
    try testing.expectEqualStrings("quantization", bs[0].key);
    try testing.expectEqualStrings("{\"bits\":4}", bs[0].json);
    try testing.expectEqualStrings("expert_quant", bs[1].key);
    try testing.expectEqual(@as(usize, 0), (try blocksOf(a, "{\"model_type\":\"llama\"}")).len);
    try testing.expectEqual(@as(usize, 0), (try blocksOf(a, "not json")).len);
}

// Bar: the refusals today's config parse makes, by name, and nothing more.
test "quant block: validation refuses what the parser refuses" {
    const dir: abi.PackDir = undefined;
    const t = struct {
        fn v(name: []const u8, block: []const u8, config: []const u8) Verdict {
            for (builtins) |b| if (std.mem.eql(u8, b.name, name)) return b.validate(.of(block), .of(config), &dir);
            unreachable;
        }
    };
    try testing.expect(t.v("affine", "{\"bits\":4,\"group_size\":64}", "{}").isOk());
    for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |bits| {
        var buf: [64]u8 = undefined;
        try testing.expect(t.v("affine", try std.fmt.bufPrint(&buf, "{{\"bits\":{d}}}", .{bits}), "{}").isOk());
    }
    try testing.expectEqual(Verdict.Code.unsupported, t.v("affine", "{\"bits\":7}", "{}").code);
    try testing.expectEqual(Verdict.Code.unsupported, t.v("affine", "{\"bits\":4,\"mode\":\"int4\"}", "{}").code);
    try testing.expectEqual(Verdict.Code.unsupported, t.v("nvfp4", "{\"bits\":4,\"mode\":\"nvfp4\",\"x.experts\":{\"mode\":\"iq2_xxs\"}}", "{}").code);
    try testing.expect(t.v("nvfp4", "{\"quant_method\":\"modelopt\"}", "{}").isOk());
    // A release's e4m3 projections inside an MXFP4 block (MiMo's fp8res residents); another fp8 grid is refused.
    try testing.expect(t.v("affine", "{\"bits\":4,\"mode\":\"mxfp4\",\"l.q_proj\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}}", "{}").isOk());
    try testing.expectEqual(Verdict.Code.unsupported, t.v("affine", "{\"bits\":4,\"mode\":\"mxfp4\",\"l.q_proj\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[128,128]}}", "{}").code);
    try testing.expect(t.v("fp8", "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", "{}").isOk());
    try testing.expectEqual(Verdict.Code.unsupported, t.v("fp8", "{\"quant_method\":\"fp8\",\"store_dtype\":\"mxfp4\"}", "{}").code);
    const prism = "{\"modules\":[{\"path\":\"a\",\"block\":1024},{\"path\":\"b\",\"block\":1024}]}";
    try testing.expect(t.v("hadamard", "{\"bits\":2,\"group_size\":128}", prism).isOk());
    try testing.expectEqual(Verdict.Code.unsupported, t.v("hadamard", "{\"bits\":2}", "{\"modules\":[{\"block\":1024},{\"block\":512}]}").code);
}

test "quant block: the declared defaults follow the block; ModelOpt is NVFP4 g16" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = declared(a, "{\"bits\":3,\"group_size\":32,\"mode\":\"affine\"}");
    try testing.expectEqual(@as(u32, 3), d.bits);
    try testing.expectEqual(@as(u32, 32), d.group_size);
    const m = declared(a, "{\"quant_method\":\"modelopt\"}");
    try testing.expectEqual(@as(u32, 16), m.group_size);
    try testing.expectEqualStrings("nvfp4", m.mode);
    try testing.expectEqual(@as(u32, 0), declared(a, "{}").bits);
    try testing.expectEqual(@as(?u32, 1024), hadamardBlock(a, "{\"modules\":[{\"block\":1024},{\"block\":1024}]}"));
    try testing.expectEqual(@as(?u32, null), hadamardBlock(a, "{\"modules\":[{\"block\":1024},{\"block\":512}]}"));
}
