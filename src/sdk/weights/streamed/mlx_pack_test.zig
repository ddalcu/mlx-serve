//! The MLX pack source (`streamed/mlx_pack.zig`, `Streamed.openPack`) over a tiny two-shard pack: every routed expert's
//! nine parts, as the slot rows hold them after a fetch, equal the tensors `mlx_load_safetensors` reads, for experts
//! stacked on axis 0 and one tensor per expert, at two affine widths, with evictions; the bill counts the slot rows,
//! not the experts; a truncated or malformed shard is refused by name. Owns tests and the pack writer.

const std = @import("std");
const build_options = @import("build_options");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const fixture = @import("../../bases/proof/fixture.zig");
const streamed = @import("../streamed.zig");

const testing = std.testing;

pub const E = 6;
const H: c_int = 128;
const I: c_int = 64;
const shard_names = [2][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
const parts = [_][]const u8{ "weight", "scales", "biases" };

/// A two-shard pack in `dir`: layer 0's experts stacked, 4-bit group 64 (gate and up in the first shard, down in the
/// second); layer 1's one tensor per expert, 8-bit group 32, in the second; a shared expert and an MTP head's experts
/// beside them (neither routed here). The index names every tensor's shard.
pub fn writePack(dir: []const u8) !void {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var shards = [2]fixture.Pack{ fixture.Pack.init(0x5a1), fixture.Pack.init(0x5a2) };
    defer for (&shards) |*p| p.deinit();
    var index: std.Io.Writer.Allocating = .init(ar);
    try index.writer.writeAll("{\"metadata\":{},\"weight_map\":{");
    var first = true;
    const W = struct {
        fn put(sh: *fixture.Pack, idx: *std.Io.Writer, fst: *bool, file: []const u8, base: []const u8, shape: []const c_int, bits: c_int, gs: c_int) !void {
            try sh.affine(base, shape, bits, gs);
            for (parts) |part| {
                try idx.print("{s}\"{s}.{s}\":\"{s}\"", .{ if (fst.*) "" else ",", base, part, file });
                fst.* = false;
            }
        }
    };
    for (projs, 0..) |proj, pi| {
        const s: usize = if (pi == 2) 1 else 0;
        const shape: [3]c_int = if (pi == 2) .{ E, H, I } else .{ E, I, H };
        try W.put(&shards[s], &index.writer, &first, shard_names[s], try std.fmt.allocPrint(ar, "model.layers.0.mlp.switch_mlp.{s}", .{proj}), &shape, 4, 64);
        try W.put(&shards[0], &index.writer, &first, shard_names[0], try std.fmt.allocPrint(ar, "mtp.layers.0.mlp.switch_mlp.{s}", .{proj}), &shape, 4, 64);
        try W.put(&shards[0], &index.writer, &first, shard_names[0], try std.fmt.allocPrint(ar, "model.layers.0.mlp.shared_expert.{s}", .{proj}), shape[1..], 4, 64);
        for (0..E) |e| try W.put(&shards[1], &index.writer, &first, shard_names[1], try std.fmt.allocPrint(ar, "model.layers.1.mlp.experts.{d}.{s}", .{ e, proj }), shape[1..], 8, 32);
    }
    try index.writer.writeAll("}}");
    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    for (shards, shard_names) |p, name| {
        const file = try std.fmt.allocPrintSentinel(ar, "{s}/{s}", .{ dir, name }, 0);
        try mlx.check(mlx.mlx_save_safetensors(file.ptr, p.map, meta));
    }
}

/// The pack's tensors as MLX reads them, every shard in one map.
const Loaded = struct {
    maps: [2]mlx.mlx_map_string_to_array,

    fn init(dir: []const u8) !Loaded {
        var l: Loaded = undefined;
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        for (&l.maps, shard_names) |*m, name| {
            var buf: [600]u8 = undefined;
            m.* = mlx.mlx_map_string_to_array_new();
            const meta = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta);
            var mt = meta;
            try mlx.check(mlx.mlx_load_safetensors(m, &mt, try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0), s));
        }
        return l;
    }

    fn deinit(l: *Loaded) void {
        for (l.maps) |m| _ = mlx.mlx_map_string_to_array_free(m);
    }

    /// Expert `e`'s component `c` of routed layer `layer` (`[out, cols]`), owned by `g`.
    fn part(l: *const Loaded, g: *ops.Graph, layer: usize, e: usize, c: usize) !mlx.mlx_array {
        var nb: [128]u8 = undefined;
        const name = if (layer == 0)
            try std.fmt.bufPrintSentinel(&nb, "model.layers.0.mlp.switch_mlp.{s}.{s}", .{ projs[c / 3], parts[c % 3] }, 0)
        else
            try std.fmt.bufPrintSentinel(&nb, "model.layers.1.mlp.experts.{d}.{s}.{s}", .{ e, projs[c / 3], parts[c % 3] }, 0);
        for (l.maps) |m| {
            var a = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_get(&a, m, name.ptr) != 0 or a.ctx == null) continue;
            const t = try g.adopt(a);
            return if (layer == 0) g.take(t, try g.fromSlice(u32, &.{@intCast(e)}, &.{1}), 0) else g.reshape(t, &.{ 1, ops.shape(t)[0], ops.shape(t)[1] });
        }
        return error.MissingWeight;
    }

    /// The bytes of every tensor whose name contains `needle`.
    fn bytes(l: *const Loaded, needles: []const []const u8) u64 {
        var n: u64 = 0;
        for (l.maps) |m| {
            const it = mlx.mlx_map_string_to_array_iterator_new(m);
            defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
            var key: ?[*:0]const u8 = null;
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            while (mlx.mlx_map_string_to_array_iterator_next(&key, &v, it) == 0) {
                const k = std.mem.span(key.?);
                if (std.mem.startsWith(u8, k, "mtp.")) continue;
                for (needles) |nd| if (std.mem.indexOf(u8, k, nd) != null) {
                    n += mlx.mlx_array_size(v) * mlx.mlx_array_itemsize(v);
                    break;
                };
            }
        }
        return n;
    }
};

fn tmpPack(tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    const dir = buf[0..try tmp.dir.realPath(std.testing.io, buf)];
    try writePack(dir);
    return dir;
}

/// Every pair of `ids` (routed layer `layer`) as its slot holds it, against the pack's tensors.
fn expectFetched(sw: *streamed.Streamed, l: *const Loaded, layer: u32, ids: []const u16) !void {
    const r = sw.routedSource();
    const f = try r.math.trunk(r.ctx, layer, ids, &.{}).get();
    var g = ops.Graph.init(testing.allocator, fixture.stream());
    defer g.deinit();
    for (ids, f.kind, f.row) |e, k, row| {
        const bk = f.banks[k].?;
        const arrs = [_]mlx.mlx_array{ bk.gate.w, bk.gate.scales, bk.gate.biases, bk.up.w, bk.up.scales, bk.up.biases, bk.down.w, bk.down.scales, bk.down.biases };
        for (arrs, 0..) |arr, c| {
            const got = try g.take(arr, try g.fromSlice(u32, &.{row}, &.{1}), 0);
            try ops.expectBitEqual(got, try l.part(&g, layer, e, c));
        }
    }
    try r.settle();
}

// ── tests ──

test "mlx pack: a fetch's slot rows hold each expert's nine parts as the pack's tensors do, stacked and per expert, two widths, two shards, with evictions" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try tmpPack(&tmp, &buf);
    var l = try Loaded.init(dir);
    defer l.deinit();
    var sw = try streamed.Streamed.openPack(testing.allocator, dir, fixture.stream(), 2);
    defer sw.close();
    try testing.expectEqual(@as(?u32, 0), sw.layerOf("model.layers.0.mlp"));
    try testing.expectEqual(@as(?u32, 1), sw.layerOf("model.layers.1.mlp"));
    try testing.expectEqual(@as(?u32, null), sw.layerOf("mtp.layers.0.mlp"));
    const calls = [_][]const u16{ &.{ 0, 5, 3, 3, 1 }, &.{ 2, 4 }, &.{ 5, 0, 1, 2, 3, 4, 4 }, &.{1} };
    for (calls) |ids| for ([_]u32{ 0, 1 }) |layer| try expectFetched(&sw, &l, layer, ids);
    const st = sw.src.pack.stats();
    try testing.expect(st.expert_cache_evictions > 0 and st.expert_cache_hits > 0);
}

test "mlx pack: streaming bills the slot rows and drops the routed experts' bytes" {
    if (!build_options.mlx_stream) return error.SkipZigTest;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try tmpPack(&tmp, &buf);
    var l = try Loaded.init(dir);
    defer l.deinit();
    const b = try streamed.Streamed.packBytes(testing.allocator, dir, 2);
    try testing.expectEqual(l.bytes(&.{ ".switch_mlp.", ".experts." }), b.experts);
    // A record is one expert's nine parts; each layer holds 2 rows, each geometry 48 transient rows.
    const rec0 = l.bytes(&.{"layers.0.mlp.switch_mlp."}) / E;
    const rec1 = l.bytes(&.{"layers.1.mlp.experts."}) / E;
    try testing.expectEqual(2 * (rec0 + rec1) + 48 * (rec0 + rec1), b.slots);
    try testing.expectEqual(b.slots, (try streamed.Streamed.packBytes(testing.allocator, dir, 2)).slots);
}

test "mlx pack: a truncated shard and a malformed header are refused by name" {
    if (!build_options.mlx_stream) return error.SkipZigTest;
    const io = std.testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try tmpPack(&tmp, &buf);
    const whole = try tmp.dir.readFileAlloc(io, shard_names[1], testing.allocator, .unlimited);
    defer testing.allocator.free(whole);
    try tmp.dir.writeFile(io, .{ .sub_path = shard_names[1], .data = whole[0 .. whole.len - 100] });
    try testing.expectError(error.StreamedPackTruncated, streamed.Streamed.packBytes(testing.allocator, dir, 2));
    var bad: [18]u8 = undefined;
    std.mem.writeInt(u64, bad[0..8], 10, .little);
    @memcpy(bad[8..], "not json!!");
    try tmp.dir.writeFile(io, .{ .sub_path = shard_names[1], .data = &bad });
    try testing.expectError(error.StreamedPackHeader, streamed.Streamed.packBytes(testing.allocator, dir, 2));
}
