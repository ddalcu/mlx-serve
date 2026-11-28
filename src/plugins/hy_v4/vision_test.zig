//! The vision tower against `tests/dump_hyv4_vision_ref.py` (user-run: HYV4_PACK, HYV4_VISION_FIXTURE).
//! Loads the real pack (its `vit.*` tensors are bf16), runs the tower over the oracle's patch pixels and
//! compares the merged image features (cos and rms ratio, the plugins/AGENTS.md parity bar).

const std = @import("std");
const testing = std.testing;
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const plugin = @import("plugin.zig");
const config = @import("config.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;

fn readAll(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
}

test "hy_v4 vision: the tower matches the reference dump (HYV4_PACK, HYV4_VISION_FIXTURE)" {
    const pack = std.c.getenv("HYV4_PACK") orelse return error.SkipZigTest;
    const fx = std.c.getenv("HYV4_VISION_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const sp = std.mem.span(pack);
    const sfx = std.mem.span(fx);

    const cfg_bytes = try readAll(a, io, sp, "config.json");
    defer a.free(cfg_bytes);
    const patches_bytes = try readAll(a, io, sfx, "patches.bin");
    defer a.free(patches_bytes);
    const features_bytes = try readAll(a, io, sfx, "features.bin");
    defer a.free(features_bytes);
    const meta_bytes = try readAll(a, io, sfx, "meta.json");
    defer a.free(meta_bytes);

    const meta = try std.json.parseFromSliceLeaky(struct { hidden: u32, out_hidden: u32, gh: u32, gw: u32, rows: u32, patches: usize }, a, meta_bytes, .{});

    var p: abi.Parsed = .{};
    const mem: @import("../../sdk/testing/fixture.zig").MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = cfg_bytes }} };
    try testing.expect(plugin.table.parse(&mem.packDir(), abi.Str.of(cfg_bytes), .{}, null, false, &p).isOk());
    defer plugin.table.free_parsed(&p);

    var fs = try FsPackDir.open(io, sp);
    defer fs.close();
    const w = try weights_load.loadWeightsOpt(io, a, sp, .{ .vision = true });
    var resident = Resident.adopt(w);
    defer resident.deinit();
    const src = resident.source();
    var why: abi.Verdict = .ok;
    const model = plugin.table.load(&.{
        .model_dir = .of(sp),
        .dir = &fs.packDir(),
        .parsed = &p,
        .weights = &src,
        .stream = .{ .ctx = null },
    }, &why);
    if (model == null) {
        std.debug.print("hy_v4 vision: load refused: {s}\n", .{why.why.slice()});
        return error.LoadRefused;
    }
    defer plugin.table.unload(model.?);

    const patches = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(patches_bytes)));
    const want = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(features_bytes)));
    const n = patches.len / meta.hidden;
    const patch_arr = mlx.mlx_array_new_data(patches.ptr, &[_]c_int{ @intCast(n), @intCast(meta.hidden) }, 2, .float32);
    defer _ = mlx.mlx_array_free(patch_arr);
    var item: abi.MediaItem = .{
        .kind = .image,
        .rows = meta.rows,
        .grid = .{ 1, meta.gh, meta.gw },
        .pixels = .{ .ctx = patch_arr.ctx },
    };
    const hooks = plugin.table.media orelse return error.NoMediaHooks;
    var out: abi.Array = .{};
    const v = hooks.tower(model.?, &item, &out);
    try testing.expect(v.isOk());
    defer _ = mlx.mlx_array_free(.{ .ctx = out.ctx });
    try ops.eval(.{ .ctx = out.ctx });
    const got_shape = mlx.mlx_array_shape(.{ .ctx = out.ctx });
    try testing.expectEqual(@as(c_int, @intCast(meta.rows)), got_shape[0]);
    try testing.expectEqual(@as(c_int, @intCast(meta.out_hidden)), got_shape[1]);
    try testing.expectEqual(want.len, meta.rows * meta.out_hidden);

    var f32_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_out);
    try mlx.check(mlx.mlx_astype(&f32_out, .{ .ctx = out.ctx }, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f32_out));
    const got = mlx.mlx_array_data_float32(f32_out) orelse return error.NoData;
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nw: f64 = 0;
    for (got[0..want.len], want) |g, w_| {
        dot += @as(f64, g) * w_;
        ng += @as(f64, g) * g;
        nw += w_ * w_;
    }
    const cos = dot / @sqrt(ng * nw);
    const rms_ratio = @sqrt(ng / nw);
    std.debug.print("[hy_v4 vision] cos {d:.6} rms_ratio {d:.6} rows {d}\n", .{ cos, rms_ratio, meta.rows });
    try testing.expect(cos >= 0.999);
    try testing.expect(@abs(rms_ratio - 1.0) <= 0.01);
}
