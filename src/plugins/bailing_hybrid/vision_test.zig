//! Oracle tests for Ling VL's tower: the SDK's `QwenVision` running the pack's `model.visual.*`
//! blocks with its merger tail loaded from the pack-root `linear_proj.*` (Geometry's merger key
//! stems), rows compared to the torch reference.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");

// ── tests ──

// Parity against a torch oracle (the release modeling file's tower math, transcription-verified
// against SGLang, run with the PACK's bf16 weights in float32). Gated on `LING_VL_FIXTURE_DIR`
// (the ~0.5 GB fixtures stay out of the repo); CI without the directory skips. bf16-vs-f32 drift is
// ~1e-3 a value on a std-0.05 signal — anything structural (wrong order, wrong table, wrong merge)
// lands at least an order of magnitude outside these bounds.
test "ling vision: the folded tower's rows match the torch oracle" {
    const dir = std.mem.span(env.get(.LING_VL_FIXTURE_DIR) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const read = struct {
        fn f(al: std.mem.Allocator, io: std.Io, d: []const u8, name: []const u8, cap: usize) ![]u8 {
            const path = try std.fs.path.join(al, &.{ d, name });
            return std.Io.Dir.cwd().readFileAlloc(io, path, al, .limited(cap));
        }
    }.f;

    const io = std.testing.io;
    const json = try read(a, io, dir, "tower_weights.json", 1 << 20);
    const bin = try read(a, io, dir, "tower_weights.bin", 1 << 31);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    var ws = Weights.init(a);
    var off: usize = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| {
        const dims_v = kv.value_ptr.array.items;
        var dims: [5]c_int = undefined;
        var n: usize = 1;
        for (dims_v, 0..) |s, i| {
            dims[i] = @intCast(s.integer);
            n *= @intCast(s.integer);
        }
        const data = bin[off..][0 .. n * 4];
        off += n * 4;
        try ws.map.put(kv.key_ptr.*, mlx.mlx_array_new_data(data.ptr, &dims, @intCast(dims_v.len), .float32));
    }
    try testing.expectEqual(off, bin.len);

    const px_bytes = try read(a, io, dir, "pixels.bin", 1 << 28);
    const rows_bytes = try read(a, io, dir, "rows.bin", 1 << 28);
    const px = mlx.mlx_array_new_data(px_bytes.ptr, &[_]c_int{ 768, 1536 }, 2, .float32);
    const want = mlx.mlx_array_new_data(rows_bytes.ptr, &[_]c_int{ 192, 2560 }, 2, .float32);

    const geom = qwen_vision.QwenVision.Geometry{
        .patch = 16, .temporal_patch = 2, .depth = 27, .hidden = 1152, .heads = 16,
        .head_dim = 72, .merge = 2, .num_pos_emb = 2304, .out_hidden = 2560,
        .merger_fc1_key = "linear_proj.0", .merger_fc2_key = "linear_proj.2",
    };
    var tower = try qwen_vision.QwenVision.init(a, geom, &ws);
    defer tower.deinit();

    const rows = try tower.forward(px, 24, 32); // [1, 192, 2560] bf16, the linear_proj tail included
    defer _ = mlx.mlx_array_free(rows);
    var got = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&got, rows, .float32, tower.s));
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_reshape(&got, got, &[_]c_int{ 192, 2560 }, 2, tower.s));

    var diff = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&diff, got, want, tower.s));
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_abs(&diff, diff, tower.s));
    var mean = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean(&mean, diff, false, tower.s));
    defer _ = mlx.mlx_array_free(mean);
    var mx = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_max(&mx, diff, false, tower.s));
    defer _ = mlx.mlx_array_free(mx);
    var mean_v: f32 = 0;
    var max_v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&mean_v, mean));
    try mlx.check(mlx.mlx_array_item_float32(&max_v, mx));
    try testing.expect(mean_v >= 0 and mean_v < 2e-3); // oracle rows have std 0.051
    try testing.expect(max_v < 3e-2);
}
