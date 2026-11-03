//! mlx-stream's NVFP4 expert bank (`mlx-stream-nvfp4-expert-bank-v1`, layout `nvfp4-gud-9part`) held resident: per
//! expert, gate / up / down as U32 codes `[out, in / 8]`, e4m3 block scales U8 `[out, in / 16]` and an f32 global
//! scale `g = weight_scale_2 x 2688`. Stacked under ModelOpt's names (`<dst>.mlp.switch_mlp.*`): each projection's
//! `weight_scale_2` beside its bank as the clamped-SwiGLU expert scales the MoE block reads.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const crt = @import("../../util/crt.zig");
const Weights = @import("../../weights/resident.zig").Weights;

const format = "mlx-stream-nvfp4-expert-bank-v1";
const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
/// ModelOpt's `weight_scale_2` is amax / (448 x 6); the bank stores amax / 6 (the manifest's `g`).
const g_over_ws2: f32 = 2688;

const Part = struct { name: []const u8, offset: u64, length: u64, shape: []const i64 };
const Record = struct { expert: u32, offset: u64 };
const Manifest = struct { format: []const u8, layout: []const u8, data_file: []const u8, num_experts: u32, parts: []const Part, records: []const Record };

/// The bank `manifest` in `dir` into `weights` under `dst` (e.g. `model.language_model.mtp.0.block`).
pub fn load(weights: *Weights, dir: []const u8, manifest: []const u8, dst: []const u8) !void {
    const a = weights.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const text = try readAll(ar, try std.fmt.allocPrintSentinel(ar, "{s}/{s}", .{ dir, manifest }, 0));
    const m = try std.json.parseFromSliceLeaky(Manifest, ar, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, m.format, format) or !std.mem.eql(u8, m.layout, "nvfp4-gud-9part")) return error.StreamBankFormat;
    if (m.records.len != m.num_experts or m.num_experts == 0) return error.StreamBankRecords;
    const fd = crt.openRead(try std.fmt.allocPrintSentinel(ar, "{s}/{s}", .{ dir, m.data_file }, 0));
    if (fd < 0) return error.StreamBankMissing;
    defer crt.close(fd);
    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);
    const E = m.num_experts;
    for (projs) |p| {
        const w = try part(m, p, "weight");
        const sc = try part(m, p, "scales");
        const gp = try part(m, p, "global_scale");
        if (w.shape.len != 2 or sc.shape.len != 2 or gp.length != 4) return error.StreamBankShape;
        const wb = try std.heap.page_allocator.alloc(u8, w.length * E);
        defer std.heap.page_allocator.free(wb);
        const sb = try std.heap.page_allocator.alloc(u8, sc.length * E);
        defer std.heap.page_allocator.free(sb);
        const ws2 = try ar.alloc(f32, E);
        for (m.records) |r| {
            if (r.expert >= E) return error.StreamBankRecords;
            try pread(fd, wb[r.expert * w.length ..][0..w.length], r.offset + w.offset);
            try pread(fd, sb[r.expert * sc.length ..][0..sc.length], r.offset + sc.offset);
            var g: [4]u8 = undefined;
            try pread(fd, &g, r.offset + gp.offset);
            ws2[r.expert] = @as(f32, @bitCast(std.mem.readInt(u32, &g, .little))) / g_over_ws2;
        }
        const wshape = [_]c_int{ @intCast(E), @intCast(w.shape[0]), @intCast(w.shape[1]) };
        const sshape = [_]c_int{ @intCast(E), @intCast(sc.shape[0]), @intCast(sc.shape[1]) };
        try put(weights, mlx.mlx_array_new_data(wb.ptr, &wshape, 3, .uint32), "{s}.mlp.switch_mlp.{s}.weight", .{ dst, p });
        try put(weights, mlx.mlx_array_new_data(sb.ptr, &sshape, 3, .uint8), "{s}.mlp.switch_mlp.{s}.scales", .{ dst, p });
        const f = mlx.mlx_array_new_data(ws2.ptr, &[_]c_int{@intCast(E)}, 1, .float32);
        defer _ = mlx.mlx_array_free(f);
        var bf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&bf, f, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        if (std.mem.eql(u8, p, "down_proj")) try put(weights, bf, "{s}.mlp.switch_mlp.expert_scale", .{dst}) else try put(weights, bf, "{s}.mlp.switch_mlp.{s}.expert_scale", .{ dst, p });
    }
}

fn part(m: Manifest, proj: []const u8, leaf: []const u8) !Part {
    for (m.parts) |p| if (std.mem.startsWith(u8, p.name, proj) and p.name.len == proj.len + 1 + leaf.len and std.mem.endsWith(u8, p.name, leaf)) return p;
    return error.StreamBankParts;
}

fn put(weights: *Weights, arr: mlx.mlx_array, comptime fmt: []const u8, args: anytype) !void {
    if (arr.ctx == null) return error.StreamBankAlloc;
    errdefer _ = mlx.mlx_array_free(arr);
    var buf: [256]u8 = undefined;
    try weights.put(try std.fmt.bufPrint(&buf, fmt, args), arr);
}

fn pread(fd: c_int, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const got = crt.pread(fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
        if (got <= 0) return error.StreamBankRead;
        done += @intCast(got);
    }
}

fn readAll(a: std.mem.Allocator, path: [:0]const u8) ![]u8 {
    const fd = crt.openRead(path);
    if (fd < 0) return error.StreamBankMissing;
    defer crt.close(fd);
    const n = crt.seekEnd(fd);
    if (n <= 0 or n > 64 << 20) return error.StreamBankManifest;
    const buf = try a.alloc(u8, @intCast(n));
    try pread(fd, buf, 0);
    return buf;
}

// ── tests ──

const testing = std.testing;

test "nvfp4 stream bank: the 9-part records stack into ModelOpt's banks and expert scales" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    // 2 experts, hidden 16, inter 16: codes [16, 2] u32 (128 B), scales [16, 1] u8 (16 B), g f32.
    const parts =
        \\[{"name":"gate_proj.weight","offset":0,"length":128,"shape":[16,2]},{"name":"gate_proj.scales","offset":128,"length":16,"shape":[16,1]},{"name":"gate_proj.global_scale","offset":144,"length":4,"shape":[]},
        \\ {"name":"up_proj.weight","offset":148,"length":128,"shape":[16,2]},{"name":"up_proj.scales","offset":276,"length":16,"shape":[16,1]},{"name":"up_proj.global_scale","offset":292,"length":4,"shape":[]},
        \\ {"name":"down_proj.weight","offset":296,"length":128,"shape":[16,2]},{"name":"down_proj.scales","offset":424,"length":16,"shape":[16,1]},{"name":"down_proj.global_scale","offset":440,"length":4,"shape":[]}]
    ;
    const man = "{\"format\":\"" ++ format ++ "\",\"layout\":\"nvfp4-gud-9part\",\"data_file\":\"b.bin\",\"num_experts\":2,\"parts\":" ++ parts ++ ",\"records\":[{\"expert\":0,\"offset\":0},{\"expert\":1,\"offset\":512}]}";
    try tmp.dir.writeFile(io, .{ .sub_path = "m.json", .data = man });
    var bin: [1024]u8 = undefined;
    for (&bin, 0..) |*b, i| b.* = @truncate(i);
    for ([_]u64{ 144, 292, 440, 512 + 144, 512 + 292, 512 + 440 }, 0..) |off, i| std.mem.writeInt(u32, bin[off..][0..4], @bitCast(@as(f32, @floatFromInt(i + 1)) * g_over_ws2), .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "b.bin", .data = &bin });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    try load(&w, dir, "m.json", "x");
    const up = w.get("x.mlp.switch_mlp.up_proj.weight").?;
    try testing.expectEqualSlices(c_int, &.{ 2, 16, 2 }, mlx.getShape(up));
    try mlx.check(mlx.mlx_array_eval(up));
    // Expert 1's up codes start at its record's offset + 148.
    try testing.expectEqual(std.mem.readInt(u32, bin[512 + 148 ..][0..4], .little), mlx.mlx_array_data_uint32(up).?[32]);
    const down = w.get("x.mlp.switch_mlp.expert_scale").?;
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    const st = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(st);
    try mlx.check(mlx.mlx_astype(&f, down, .float32, st));
    try mlx.check(mlx.mlx_array_eval(f));
    try testing.expectEqualSlices(f32, &.{ 3, 6 }, mlx.mlx_array_data_float32(f).?[0..2]);
}
