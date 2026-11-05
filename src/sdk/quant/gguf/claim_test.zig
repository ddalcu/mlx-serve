//! `claim.zig`'s real-pack test: it parses the config and tokenizer the claimed GGUF stands in for, through the engine.

const std = @import("std");
const env = @import("../../../env.zig");
const claim = @import("claim.zig");
const claimState = claim.claimState;
const servablePath = claim.servablePath;
const weightBytes = claim.weightBytes;

test "real GGUF: stands in for a model dir, and --engine turns it off (a real GGUF, env-gated)" {
    const model = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const model_mod = @import("../../config/model.zig");
    const tokenizer_mod = @import("../../tokenizer/tokenizer.zig");
    const prev = claimState().enabled;
    defer claimState().enabled = prev;
    claimState().enabled = true;

    var config = try model_mod.parseConfig(io, allocator, model);
    defer config.deinit(allocator);
    try std.testing.expectEqual(model_mod.QuantMode.gguf, config.quant_mode);
    try std.testing.expect(config.num_hidden_layers > 0 and config.quant_bits > 0);

    var tok = try tokenizer_mod.loadTokenizer(io, allocator, model);
    defer tok.deinit();
    const text = "Hello, GGUF on MLX! 12345 héllo";
    const ids = try tok.encode(allocator, text);
    defer allocator.free(ids);
    const back = try tok.decode(allocator, ids, false);
    defer allocator.free(back);
    try std.testing.expectEqualStrings(text, back);
    try std.testing.expect(tok.special_tokens.count() > 0);

    try std.testing.expect(weightBytes(io, model).? > 1 << 20);
    claimState().enabled = false;
    try std.testing.expectEqual(@as(?[]u8, null), servablePath(io, allocator, model));
}

test "real GGUF: a DeepSeek-V4.1 set's DSpark bundle beside it maps and serves every tensor (a real GGUF, env-gated)" {
    const model = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const root = @import("root.zig");
    const ds41 = root.gguf_arch.deepseek41;
    const allocator = std.testing.allocator;
    var f = try root.gguf.File.open(allocator, model);
    defer f.deinit();
    const arch = try root.gguf_arch.Arch.read(&f);
    if (arch != .deepseek41) return error.SkipZigTest;
    const head = @import("files.zig").findMtpSidecar(std.testing.io, allocator, model) orelse return error.SkipZigTest;
    defer allocator.free(head);
    var h = try root.gguf.File.open(allocator, head);
    defer h.deinit();
    const ds = try ds41.Dspark.read(&h, arch.deepseek41);
    var buf: [160]u8 = undefined;
    for (h.tensors.keys(), h.tensors.values()) |name, t| {
        const m = ds41.mapDsparkTensor(&buf, name, ds) orelse {
            std.debug.print("unmapped {s}\n", .{name});
            return error.TestUnexpectedResult;
        };
        try std.testing.expect(arch.serves(m.name, t.ty));
    }
    try std.testing.expect(try root.weights.dsparkResidentBytes(&h, arch.deepseek41) > 1 << 30);
}

test "a DeepSeek-V4.1 set's rebuilt config with its vision GGUF's tower parses in the engine's config into the release's tower" {
    const ds41 = @import("root.zig").gguf_arch.deepseek41;
    const allocator = std.testing.allocator;
    var f = try ds41.testFileDs4(ds41.release_config);
    defer f.deinit();
    var v = try ds41.testVisionFileDs4();
    defer v.deinit();
    const hp = try ds41.Hparams.read(&f);
    const json = try ds41.configJsonWith(allocator, hp, null, try ds41.Vision.read(&v, hp, null));
    defer allocator.free(json);
    var config = try @import("../../config/model.zig").parseConfigFromJson(allocator, json);
    defer config.deinit(allocator);
    try std.testing.expect(config.dsv41_vision and config.has_vision);
    try std.testing.expectEqual(@as(u32, 129264), config.image_token_id);
    try std.testing.expectEqual(@as(u32, 32), config.vision_num_layers);
    try std.testing.expectEqual(@as(u32, 1024), config.vision_hidden_size);
    try std.testing.expectEqual(@as(u32, 2816), config.vision_intermediate_size);
    try std.testing.expectEqual(@as(u32, 16), config.vision_num_heads);
    try std.testing.expectEqual(@as(u32, 14), config.vision_patch_size);
    try std.testing.expectEqual(@as(u32, 295936), config.dv_min_pixels);
}

test "real GGUF: a DeepSeek-V4.1 set's vision GGUF beside it maps every tensor and declares the tower (a real GGUF, env-gated)" {
    const model = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const root = @import("root.zig");
    const ds41 = root.gguf_arch.deepseek41;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var f = try root.gguf.File.open(allocator, model);
    defer f.deinit();
    const arch = try root.gguf_arch.Arch.read(&f);
    if (arch != .deepseek41) return error.SkipZigTest;
    const tower = @import("files.zig").findVisionSidecar(io, allocator, model) orelse return error.SkipZigTest;
    defer allocator.free(tower);
    var v = try root.gguf.File.open(allocator, tower);
    defer v.deinit();
    const vis = try ds41.Vision.read(&v, arch.deepseek41, f.getString("general.source.revision"));
    try std.testing.expectEqual(@as(u32, 32), vis.layers);
    var mbuf: [96]u8 = undefined;
    for (v.tensors.keys(), v.tensors.values()) |name, t| {
        const m = ds41.mapVisionTensor(&mbuf, name) orelse {
            std.debug.print("unmapped {s}\n", .{name});
            return error.TestUnexpectedResult;
        };
        try std.testing.expect(arch.serves(m.name, t.ty));
    }
    try std.testing.expect(try root.weights.visionResidentBytes(&v, arch.deepseek41) > 900 << 20);

    const prev = claimState().enabled;
    defer claimState().enabled = prev;
    claimState().enabled = true;
    var config = try @import("../../config/model.zig").parseConfig(io, allocator, model);
    defer config.deinit(allocator);
    try std.testing.expect(config.dsv41_vision and config.image_token_id == 129264 and config.vision_num_layers == 32);
}
