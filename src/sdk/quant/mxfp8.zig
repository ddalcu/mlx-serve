//! sdk/quant/mxfp8.
//! Owns lmHeadMxfp8On, quantizeHeadMxfp8.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Weights = @import("../weights/resident.zig").Weights;

/// LOSSY, so on by default only where it was measured: Flash Next (`qwen4`) on an integrated
/// CUDA GPU (DGX Spark), for a quantized pack whose head ships dense. A bf16 pack keeps its reference head.
pub fn lmHeadMxfp8On(setting: ?bool, cli: ?bool, cuda_integrated: bool, qwen4: bool, quant_bits: u32) bool {
    return setting orelse cli orelse (cuda_integrated and qwen4 and quant_bits > 0);
}

/// Quantize the dense head `w` (the map's `key`) to MXFP8 g32, the bytes vLLM serves under
/// `VLLM_MXFP8_LM_HEAD`, and hand the map the packed weight and its scales in place of the
/// bf16 copy, which is freed. Returns the scales, borrowed from the map.
pub fn quantizeHeadMxfp8(weights: *Weights, key: []const u8, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, s));
    try mlx.check(mlx.mlx_eval(pair)); // before the bf16 copy goes
    var qw = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qw);
    var qs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qs);
    try mlx.check(mlx.mlx_vector_array_get(&qw, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qs, pair, 1));
    var name: [176]u8 = undefined;
    try weights.put(try std.fmt.bufPrint(&name, "{s}.scales", .{key[0 .. key.len - ".weight".len]}), qs);
    weights.replace(key, qw);
    return qs;
}

// ── tests ──

const model_mod = @import("../config/model.zig");
const plugin_driver = @import("../../engines/mlx/plugin/driver.zig");

// Bar: the default is on only where measured (Flash Next, quantized pack, integrated CUDA); a flag beats it, a model setting beats both.
test "lm_head mxfp8: on by default for quantized Flash Next packs on integrated CUDA, settings and flags win" {
    const t = std.testing;
    try t.expect(lmHeadMxfp8On(null, null, true, true, 4));
    try t.expect(!lmHeadMxfp8On(null, null, true, false, 4));
    try t.expect(!lmHeadMxfp8On(null, null, true, true, 0));
    try t.expect(!lmHeadMxfp8On(null, null, false, true, 4));
    try t.expect(!lmHeadMxfp8On(null, false, true, true, 4));
    try t.expect(lmHeadMxfp8On(null, true, false, false, 0));
    try t.expect(lmHeadMxfp8On(true, false, false, false, 0));
    try t.expect(!lmHeadMxfp8On(false, true, true, true, 4));
}

// Bar: vLLM's MXFP8 bytes (scale 2^ceil(log2(amax/448)), e4m3 nearest-even) behind the served head, logits within 5% RMS.
test "lm_head mxfp8: a dense head loads as vLLM's MXFP8 bytes in place of the bf16 copy" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    const Fix = @import("../../engines/mlx/spec/dflash/dflash.zig").TinyFix;
    const V = Fix.VOCAB;
    const H = Fix.HIDDEN;
    const E4m3 = struct {
        fn nearestEven(x: f32) f32 {
            const a = @abs(x);
            if (a == 0) return 0;
            const step = std.math.exp2(@max(@floor(std.math.log2(a)), -6) - 3);
            const r = @floor(a / step);
            const up = a / step - r > 0.5 or (a / step - r == 0.5 and @mod(r, 2) == 1);
            return std.math.copysign((if (up) r + 1 else r) * step, x);
        }
    };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try Fix.writeTrunk(io, tmp.dir, dir, s);
    // An untied pack: its own bf16 head beside the trunk.
    {
        const untied = try std.mem.replaceOwned(u8, allocator, Fix.TRUNK_CONFIG, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false");
        defer allocator.free(untied);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = untied });
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const head = try Fix.bf16Arr(V, H, 99, s);
        defer _ = mlx.mlx_array_free(head);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, "lm_head.weight", head));
        const file = try std.fmt.allocPrintSentinel(allocator, "{s}/head.safetensors", .{dir}, 0);
        defer allocator.free(file);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
    }
    const ids = mlx.mlx_array_new_data(&[_]i32{ 1, 4, 7 }, &[_]c_int{ 1, 3 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    var logits: [2][]f32 = .{ &.{}, &.{} };
    defer for (logits) |l| allocator.free(l);
    for ([_]bool{ false, true }, &logits) |mxfp8, *out| {
        var config = try model_mod.parseConfig(io, allocator, dir);
        defer config.deinit(allocator);
        config.lm_head_mxfp8_override = mxfp8;
        var xfm = try plugin_driver.load(io, allocator, &config, dir, mlx.gpuStream());
        defer xfm.deinit();
        const head = xfm.lm_head_w;
        try t.expectEqual(mxfp8, xfm.lm_head_s.ctx != null);
        if (mxfp8) {
            try t.expect(mlx.mlx_array_dtype(xfm.lm_head_s) == .uint8);
            try t.expectEqualSlices(c_int, &.{ V, H / 32 }, mlx.getShape(xfm.lm_head_s));
            const ref = try Fix.bf16Arr(V, H, 99, s);
            defer _ = mlx.mlx_array_free(ref);
            const w = try Fix.readF32(ref, allocator, s);
            defer allocator.free(w);
            var deq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq);
            try mlx.check(mlx.mlx_dequantize(&deq, head, xfm.lm_head_s, .{ .ctx = null }, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
            const got = try Fix.readF32(deq, allocator, s);
            defer allocator.free(got);
            try mlx.check(mlx.mlx_array_eval(xfm.lm_head_s));
            const bytes = mlx.mlx_array_data_uint8(xfm.lm_head_s).?;
            for (0..V * H / 32) |g| {
                var amax: f32 = 0;
                for (w[g * 32 ..][0..32]) |x| amax = @max(amax, @abs(x));
                const e = @ceil(std.math.log2(amax / 448.0));
                try t.expectEqual(@as(u8, @intFromFloat(e + 127)), bytes[g]);
                for (w[g * 32 ..][0..32], got[g * 32 ..][0..32]) |x, y|
                    try t.expectEqual(E4m3.nearestEven(x / std.math.exp2(e)) * std.math.exp2(e), y);
            }
        }
        const l = try xfm.forward(ids);
        defer _ = mlx.mlx_array_free(l);
        out.* = try Fix.readF32(l, allocator, s);
    }
    var err: f64 = 0;
    var ref: f64 = 0;
    for (logits[0], logits[1]) |a, b| {
        err += (b - a) * (b - a);
        ref += a * a;
    }
    try t.expect(err > 0 and @sqrt(err / ref) < 0.05);
}
