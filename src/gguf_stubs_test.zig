const std = @import("std");
const mlx = @import("mlx_stub.zig");
const transformer = @import("transformer_stub.zig");
const diffusion = @import("diffusion_stub.zig");
const weights = @import("model_weights_stub.zig");

test "gguf-only: MLX telemetry reports no device or allocations" {
    try std.testing.expect(mlx.noGpuBackend());
    var available = true;
    try mlx.check(mlx.mlx_metal_is_available(&available));
    try std.testing.expect(!available);
    var bytes: usize = 123;
    try mlx.check(mlx.mlx_get_active_memory(&bytes));
    try std.testing.expectEqual(@as(usize, 0), bytes);
    bytes = 123;
    try mlx.check(mlx.mlx_get_cache_memory(&bytes));
    try std.testing.expectEqual(@as(usize, 0), bytes);
    bytes = 123;
    try mlx.check(mlx.mlx_get_peak_memory(&bytes));
    try std.testing.expectEqual(@as(usize, 0), bytes);
}

test "gguf-only: embedded slots accept an empty cache and refuse MLX layers" {
    var cache = try transformer.KVCache.initWithConfigAndHeadDim(std.testing.allocator, 0, transformer.KVQuantConfig.dense, 128);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 0), cache.step);
    try std.testing.expectError(error.MlxUnavailable, transformer.KVCache.initWithConfigAndHeadDim(std.testing.allocator, 1, transformer.KVQuantConfig.dense, 128));
    try std.testing.expectError(error.MlxUnavailable, cache.reinit(1, transformer.KVQuantConfig.dense, 128));
}

test "gguf-only: unsupported loaders refuse before accessing model files" {
    try std.testing.expectError(error.MlxUnavailable, weights.loadWeightsSingleFile(std.testing.allocator, "/nonexistent/model.safetensors"));
    try std.testing.expectError(error.MlxUnavailable, diffusion.Runner.init(std.testing.allocator, {}, {}, {}, {}));
}

test "gguf-only: shared multipart translation preserves data and rejects numeric injection" {
    const media = @import("gen_common.zig");
    const a = std.testing.allocator;
    const prefix = "--test\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit this\r\n" ++
        "--test\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\n\r\nPNG\r\n" ++
        "--test\r\nContent-Disposition: form-data; name=\"steps\"\r\n\r\n";
    const json = try media.openaiEditFormToJson(a, prefix ++ "8\r\n--test--\r\n", "multipart/form-data; boundary=test");
    defer a.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("edit this", parsed.value.object.get("prompt").?.string);
    try std.testing.expectEqualStrings("UE5H", parsed.value.object.get("image").?.string);
    try std.testing.expectEqual(@as(i64, 8), parsed.value.object.get("steps").?.integer);
    try std.testing.expectError(error.MalformedNumber, media.openaiEditFormToJson(a, prefix ++ "8,\"stream\":true\r\n--test--\r\n", "multipart/form-data; boundary=test"));
}

test "gguf-only: media peak estimates take explicit backend inputs" {
    const media = @import("gen_common.zig");
    try std.testing.expect(!(media.PeakBackend{}).adaln_precompute());
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "text_encoder", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "text_encoder/weights.safetensors", .data = "1234" });
    try tmp.dir.writeFile(io, .{ .sub_path = "dit.safetensors", .data = "123456" });
    const probe = struct {
        fn workingSet() u64 {
            return 8 << 30;
        }
        fn tower(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
            return true;
        }
    };
    const backend: media.PeakBackend = .{ .working_set = probe.workingSet, .tower_present = probe.tower };
    const expected = media.qwenImageEditPeakBytes(4, 6, 8 << 30, true);
    try std.testing.expectEqual(expected, media.estimatePeakResidentBytesInDir(io, tmp.dir, "qwen_image", "/model", backend));
    try std.testing.expectEqual(@as(u64, 10), media.estimatePeakResidentBytesIn(io, tmp.dir, "other"));
    try std.testing.expectEqual(@as(u64, 10) + media.MUSIC3_GEN_BUFFER_BYTES, media.estimatePeakResidentBytesIn(io, tmp.dir, "minimax_music3"));
}

test "gguf-only: shared KV settings preserve the wire vocabulary" {
    const cfg = @import("kv_quant_config.zig").KVQuantConfig;
    try std.testing.expectEqualStrings("off", cfg.fromJsonValue(.{ .integer = 0 }).?.wireName());
    try std.testing.expectEqualStrings("4", cfg.fromJsonValue(.{ .string = "4" }).?.wireName());
    try std.testing.expectEqualStrings("8", cfg.fromJsonValue(.{ .integer = 8 }).?.wireName());
    try std.testing.expect(cfg.fromJsonValue(.{ .integer = 3 }) == null);
    try std.testing.expect(cfg.fromJsonValue(.null) == null);
}
