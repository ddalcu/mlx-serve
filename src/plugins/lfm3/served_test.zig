//! The lfm3 pack served through the engine: `model.parseConfig` (the plugin's facts, the legacy parse having no lfm3
//! arm), the driver's load, and the engine's forward against mlx-vlm's logits (`tests/dump_lfm3_fixtures.py`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const parity = @import("../../sdk/testing/parity.zig");

const testing = std.testing;
const VOCAB = 128;
const prompt = [_]i32{ 1, 17, 42, 125, 3, 99, 7, 80, 64, 5, 122, 31 };

fn stage(io: std.Io, tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/tiny/config.json") });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = @embedFile("fixtures/tiny/model.safetensors") });
    return buf[0..try tmp.dir.realPath(io, buf)];
}

test "lfm3 served: the engine config is the plugin's facts, conv and sliding layers included" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    var cfg = try model.parseConfig(io, a, try stage(io, &tmp, &path_buf));
    defer cfg.deinit(a);
    try testing.expect(cfg.plugin != null);
    try testing.expectEqualStrings("lfm3", cfg.model_type);
    try testing.expectEqualStrings("language_model.model", cfg.weight_prefix);
    try testing.expect(cfg.has_hybrid_layers and cfg.has_sliding_window);
    try testing.expectEqual(@as(u32, 4), cfg.sliding_window);
    try testing.expect(cfg.layer_is_global[1] and !cfg.layer_is_global[2]);
    try testing.expectEqual(@as(u32, 16), cfg.head_dim);
}

test "lfm3 served: the driver's forward of the prompt matches mlx-vlm's LFM3" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = try stage(io, &tmp, &path_buf);
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    var xfm = try driver.load(io, a, &cfg, dir, s);
    defer xfm.deinit();

    const ids = mlx.mlx_array_new_data(&prompt, &[_]c_int{ 1, prompt.len }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try xfm.forward(ids);
    defer _ = mlx.mlx_array_free(logits);
    const ours = try parity.qwen4ReadF32(a, logits, s);
    defer a.free(ours);
    const bytes = @embedFile("fixtures/tiny/prefill_logits.f32");
    const ref = @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
    const cmp = parity.qwen4CompareRows(ours, ref, prompt.len, VOCAB);
    // The bar of proof_test.zig: bf16 activations against the f32 reference.
    try testing.expect(cmp.min_cos > 0.999 and parity.decidedMisses(ours, ref, prompt.len, VOCAB) == 0);
}

test "lfm3 codec: a whole decode matches mlx-vlm's, and a stream in chunks equals the whole (LFM3_AUDIO_MODEL, LFM3_CODEC_FIXTURE)" {
    // tests/dump_lfm3_fixtures.py --codec: 40 frames of seeded codes and the reference's f32 waveform.
    const pack = std.c.getenv("LFM3_AUDIO_MODEL") orelse return error.SkipZigTest;
    const fx_dir = std.c.getenv("LFM3_CODEC_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const codec = @import("codec.zig");
    const reader = @import("../../sdk/bases/reader.zig");
    var fs = try @import("../../sdk/util/pack_dir.zig").FsPackDir.open(io, std.mem.span(pack));
    defer fs.close();
    const pd = fs.packDir();
    const parsed: @import("../../abi/plugin.zig").Parsed = .{};
    const ctx: @import("../../abi/plugin.zig").LoadCtx = .{ .model_dir = .of(std.mem.span(pack)), .dir = &pd, .parsed = &parsed, .stream = .{ .ctx = mlx.gpuStream().ctx } };
    var owned: reader.Owned = .{};
    defer owned.deinit();
    var c = (try codec.Codec.load(&ctx, &owned)).?;
    defer c.deinit();

    var path: [1024]u8 = undefined;
    const read = struct {
        fn f(al: std.mem.Allocator, rio: std.Io, p: []const u8) ![]u8 {
            return std.Io.Dir.cwd().readFileAlloc(rio, p, al, .limited(1 << 24));
        }
    }.f;
    const code_bytes = try read(a, io, try std.fmt.bufPrint(&path, "{s}/codes.u32", .{std.mem.span(fx_dir)}));
    defer a.free(code_bytes);
    const wave_bytes = try read(a, io, try std.fmt.bufPrint(&path, "{s}/wave.f32", .{std.mem.span(fx_dir)}));
    defer a.free(wave_bytes);
    const codes = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(code_bytes)));
    const want = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(wave_bytes)));

    const whole = try c.decode(a, codes);
    defer a.free(whole);
    try testing.expectEqual(want.len, whole.len);
    const cmp = parity.qwen4CompareRows(whole, want, 1, want.len);
    std.debug.print("[lfm3 codec] whole vs mlx-vlm f32: cos {d:.6}\n", .{cmp.min_cos});
    // The reference's own bf16 decode sits at 0.9989; the trunk amplifies its input's rounding.
    try testing.expect(cmp.min_cos > 0.998);

    // Frames pushed in uneven chunks past the decoder's left context: the same samples, each final when no later
    // frame reaches it.
    var st = try codec.Stream.init(a, &c);
    defer st.deinit();
    var out: std.ArrayList(f32) = .empty;
    defer out.deinit(a);
    try testing.expect(c.context < 25);
    for ([_][2]usize{ .{ 0, 3 }, .{ 3, 7 }, .{ 7, 25 }, .{ 25, 26 }, .{ 26, 40 } }) |r| try st.push(codes[r[0] * 12 .. r[1] * 12], &out);
    try st.finish(&out);
    try testing.expectEqual(whole.len, out.items.len);
    const scmp = parity.qwen4CompareRows(out.items, want, 1, want.len);
    var worst: f32 = 0;
    var peak: f32 = 0;
    for (out.items, whole) |x, y| {
        worst = @max(worst, @abs(x - y));
        peak = @max(peak, @abs(y));
    }
    std.debug.print("[lfm3 codec] stream vs mlx-vlm f32: cos {d:.6}; vs whole max |diff| {e} (peak {e})\n", .{ scmp.min_cos, worst, peak });
    try testing.expect(scmp.min_cos > 0.998);
}
