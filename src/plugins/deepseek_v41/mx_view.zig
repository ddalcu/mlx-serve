//! An exact MXFP4 view of an NVFP4 routed-expert bank, built at load (`MLX_SERVE_DSV41_MXFP4_VIEW`). A 32-value span
//! whose two e4m3 scales are one positive power of two 2^a, under its expert's global 2^b, holds the values of the
//! MXFP4 group of the same e2m1 codes under the e8m0 scale 2^(a + b): MLX packs both formats' codes alike, so only the
//! scales are rebuilt (`scripts/dsv41_nvfp4_pack.py --experts-mxfp4` writes the same bytes offline).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const dsv41 = @import("deepseek_v41.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;

pub fn on() bool {
    return env.on(.MLX_SERVE_DSV41_MXFP4_VIEW);
}

/// The exponent of e4m3 byte `b` where it is a positive power of two: 2^-9 .. 2^-7 subnormal, 2^-6 .. 2^8 normal.
pub fn e4m3Exponent(b: u8) ?i16 {
    if (b & 0x80 != 0) return null;
    const e: i16 = b >> 3;
    const m = b & 7;
    if (e == 0) return switch (m) {
        1 => -9,
        2 => -8,
        4 => -7,
        else => null,
    };
    return if (m == 0) e - 7 else null;
}

/// e8m0's normal codes, 2^(c - 127): MLX reads 0 as the subnormal 2^-127, whose products the GPU flushes to zero.
const code_lo: i16 = 1;
const code_hi: i16 = 254;
/// A scale pair that is no single power of two: below `code_lo` whatever global lifts it.
const no_pair: i16 = -1024;

/// Why a bank stays NVFP4.
pub const Refusal = union(enum) {
    /// Expert `expert`'s global is not a positive power of two.
    global: struct { expert: usize, value: f32 },
    /// Spans whose two e4m3 scales are not one positive power of two, or whose code with the global is not a normal
    /// e8m0 code.
    spans: u64,
};

pub const View = union(enum) { mx: A, nv: Refusal };

/// The e8m0 scales `[E, out, in/32]` of the MXFP4 bank that holds the values of the NVFP4 bank with e4m3 scales `nv`
/// (`[E, out, in/16]` u8) under the per-expert globals `global` (`[E]` bf16), evaluated (caller frees); else why the
/// bank stays NVFP4.
pub fn scales(gpa: std.mem.Allocator, s: S, nv: A, global: A) !View {
    const sh = mlx.getShape(nv);
    if (sh.len != 3 or @rem(sh[2], 2) != 0 or mlx.mlx_array_dtype(nv) != .uint8 or mlx.mlx_array_size(global) != @as(usize, @intCast(sh[0])))
        return error.Dsv41MxViewLayout;
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    // Each expert's global 2^b as its code offset 127 + b.
    const g = try sc.astype(global, .float32);
    try mlx.check(mlx.mlx_array_eval(g));
    const host = (mlx.mlx_array_data_float32(g) orelse return error.NoData)[0..mlx.mlx_array_size(g)];
    const off = try gpa.alloc(i16, host.len);
    defer gpa.free(off);
    for (host, off, 0..) |x, *o, i| {
        const f = std.math.frexp(x);
        if (!std.math.isFinite(x) or x <= 0 or f.significand != 0.5) return .{ .nv = .{ .global = .{ .expert = i, .value = x } } };
        o.* = @intCast(126 + f.exponent);
    }
    // A span's two scale bytes, read as one u16, to their exponent a (or `no_pair`).
    const lut = try gpa.alloc(i16, 1 << 16);
    defer gpa.free(lut);
    for (lut, 0..) |*v, p| {
        const lo: u8 = @truncate(p);
        v.* = if (lo == @as(u8, @truncate(p >> 8))) e4m3Exponent(lo) orelse no_pair else no_pair;
    }
    const lut_a = try sc.keep(mlx.mlx_array_new_data(lut.ptr, &[_]c_int{1 << 16}, 1, .int16));
    const off_a = try sc.keep(mlx.mlx_array_new_data(off.ptr, &[_]c_int{ sh[0], 1, 1 }, 3, .int16));
    const k = try sc.add(try sc.op2(mlx.mlx_take, lut_a, try sc.view(nv, .uint16)), off_a);
    const flat = try sc.reshape(k, &.{-1});
    const least = try sc.reduce(mlx.mlx_min_axis, flat, 0, false);
    const most = try sc.reduce(mlx.mlx_max_axis, flat, 0, false);
    const fits = try sc.op2(mlx.mlx_logical_and, try sc.op2(mlx.mlx_greater_equal, least, try int16(&sc, code_lo)), try sc.op2(mlx.mlx_less_equal, most, try int16(&sc, code_hi)));
    const mx = try sc.astype(k, .uint8);
    const outs = mlx.mlx_vector_array_new_data(&[_]A{ mx, fits }, 2);
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_eval(outs));
    var whole = false;
    try mlx.check(mlx.mlx_array_item_bool(&whole, fits));
    if (whole) return .{ .mx = sc.out(mx) };
    const off_code = try sc.op2(mlx.mlx_logical_or, try sc.op2(mlx.mlx_less, k, try int16(&sc, code_lo)), try sc.op2(mlx.mlx_greater, k, try int16(&sc, code_hi)));
    var n = mlx.mlx_array_new();
    const bad = try sc.res(mlx.mlx_sum(&n, try sc.astype(off_code, .int32), false, s), &n);
    try mlx.check(mlx.mlx_array_eval(bad));
    var spans: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&spans, bad));
    return .{ .nv = .{ .spans = @intCast(spans) } };
}

fn int16(sc: *dsv41.Scope, v: i16) !A {
    return sc.keep(mlx.mlx_array_new_data(&v, &[_]c_int{}, 0, .int16));
}

/// The warning that names a bank `why` keeps NVFP4; `proj` 0, 1, 2 is w1, w3, w2.
pub fn refused(bank: []const u8, proj: usize, why: Refusal) void {
    const w = ([_][]const u8{ "w1", "w3", "w2" })[proj];
    switch (why) {
        .global => |g| log.warn("deepseek_v41: {s} stays NVFP4 (MLX_SERVE_DSV41_MXFP4_VIEW): expert {d}'s {s} global {e} is not a power of two\n", .{ bank, g.expert, w, g.value }),
        .spans => |n| log.warn("deepseek_v41: {s} stays NVFP4 (MLX_SERVE_DSV41_MXFP4_VIEW): {d} of {s}'s 32-value spans are not one power of two inside e8m0's normal codes\n", .{ bank, n, w }),
    }
}

/// What one load's view did, for its one log line.
pub const Tally = struct {
    viewed: u32 = 0,
    refused: u32 = 0,
    /// The e4m3 scales and globals let go, and the e8m0 scales in their place.
    nv_bytes: u64 = 0,
    mx_bytes: u64 = 0,
    ms: i64 = 0,

    pub fn report(t: *const Tally) void {
        if (t.viewed + t.refused == 0) return;
        log.info("deepseek_v41: NVFP4 routed experts served as MXFP4 (MLX_SERVE_DSV41_MXFP4_VIEW): {d} of {d} banks, {d:.2} GB of e4m3 scales and globals as {d:.2} GB of e8m0, {d} ms{s}\n", .{ t.viewed, t.viewed + t.refused, gb(t.nv_bytes), gb(t.mx_bytes), t.ms, if (t.refused > 0) "; the others stay NVFP4, named above" else "" });
    }
};

fn gb(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// The bytes a load with the view holds below the pack's shards: half of each NVFP4 expert bank's e4m3 scales (an
/// e8m0 code is one byte a 32-span, its e4m3 pair two) and all its globals, for every bank the index names globals for
/// (the stages' only when they load). A bank the load keeps NVFP4 is named in the log, not foreseen here.
pub fn savedBytes(gpa: std.mem.Allocator, dir_path: []const u8, dspark: bool) !u64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{}) catch |e| return if (e == error.NotDir) 0 else e;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch return 0;
    const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return 0;
    const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return 0;
    if (wm != .object) return 0;
    // The banks with globals, as their name prefix (`layers.L.ffn.experts.`).
    const tail = "gate_proj.expert_scale";
    var banks: std.StringArrayHashMapUnmanaged(void) = .empty;
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (wm.object.keys(), wm.object.values()) |key, v| {
        if (v == .string) try files.put(a, v.string, {});
        const bare = bareKey(key);
        if (std.mem.endsWith(u8, bare, ".ffn.experts." ++ tail) and (dspark or !std.mem.startsWith(u8, bare, "mtp.")))
            try banks.put(a, bare[0 .. bare.len - tail.len], {});
    }
    var total: u64 = 0;
    for (files.keys()) |name| {
        const f = dir.openFile(io, name, .{}) catch continue;
        defer f.close(io);
        var rb: [8192]u8 = undefined;
        var rs = f.reader(io, &rb);
        const len = rs.interface.takeInt(u64, .little) catch continue;
        if (len > 64 << 20) continue;
        const hdr = try a.alloc(u8, @intCast(len));
        rs.interface.readSliceAll(hdr) catch continue;
        const h = std.json.parseFromSliceLeaky(std.json.Value, a, hdr, .{}) catch continue;
        if (h != .object) continue;
        var it = h.object.iterator();
        while (it.next()) |e| {
            const owner = wm.object.get(e.key_ptr.*) orelse continue;
            if (owner != .string or !std.mem.eql(u8, owner.string, name)) continue;
            const bare = bareKey(e.key_ptr.*);
            const at = std.mem.indexOf(u8, bare, ".ffn.experts.") orelse continue;
            if (!banks.contains(bare[0 .. at + ".ffn.experts.".len])) continue;
            const scale = std.mem.endsWith(u8, bare, ".scales");
            if (!scale and !std.mem.endsWith(u8, bare, "expert_scale")) continue;
            const off = (if (e.value_ptr.* == .object) e.value_ptr.object.get("data_offsets") else null) orelse continue;
            if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer) continue;
            const bytes: u64 = @intCast(@max(off.array.items[1].integer - off.array.items[0].integer, 0));
            total += if (scale) bytes / 2 else bytes;
        }
    }
    return total;
}

fn bareKey(key: []const u8) []const u8 {
    const p = weights_load.dsv41_text_prefix;
    return if (std.mem.startsWith(u8, key, p)) key[p.len..] else key;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("decode/fixture.zig");
const resident = @import("../../sdk/weights/resident.zig");
const decode = @import("decode/decode.zig");
const Config = @import("config.zig").Config;
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const none: A = .{ .ctx = null };

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// `a`'s bytes, row-major (caller frees).
fn bytesOf(a: A) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return testing.allocator.dupe(u8, p[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)]);
}

fn u8Array(bytes: []const u8, shape: []const c_int) A {
    return mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), .uint8);
}

fn bf16Array(v: []const f32) !A {
    const f = mlx.mlx_array_new_data(v.ptr, &[_]c_int{ci(v.len)}, 1, .float32);
    defer _ = mlx.mlx_array_free(f);
    var b = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&b, f, .bfloat16, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(b));
    return b;
}

/// The converter's e4m3 read (`e4m3_lut`): NaN at S.1111.111.
fn e4m3F64(b: u8) f64 {
    if (b & 0x7F == 0x7F) return std.math.nan(f64);
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    const v = if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
    return if (b & 0x80 != 0) -v else v;
}

/// `dsv41_nvfp4_pack.py`'s `mx_scales` for a span whose two halves both hold a nonzero code: each half's e4m3 scale
/// times the global, in float64, a power of two 2^k (frexp's mantissa 0.5), the two k equal, k in [-127, 127]; the
/// code k + 127.
fn converterCode(pair: [2]u8, global: f32) ?u8 {
    var k: [2]i32 = undefined;
    for (pair, &k) |b, *x| {
        const f = std.math.frexp(e4m3F64(b) * @as(f64, global));
        if (f.significand != 0.5) return null;
        x.* = f.exponent - 1;
    }
    if (k[0] != k[1] or k[0] < -127 or k[0] > 127) return null;
    return @intCast(k[0] + 127);
}

/// The e4m3 byte of 2^a, a in [-9, 8].
fn e4m3Of(a: i32) u8 {
    return switch (a) {
        -9 => 0x01,
        -8 => 0x02,
        -7 => 0x04,
        else => @intCast((a + 7) << 3),
    };
}

test "dsv41 mxfp4 view: a span is an MXFP4 group exactly where its two e4m3 scales are one positive power of two and its e8m0 code is normal, coded as the converter codes it" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    // Every positive power of two e4m3 holds, paired, under two globals: the converter's codes.
    var pow2: [18]u8 = undefined;
    var np: usize = 0;
    for (0..256) |b| if (e4m3Exponent(@intCast(b)) != null) {
        pow2[np] = @intCast(b);
        np += 1;
    };
    try testing.expectEqual(@as(usize, 18), np);
    const gl = [2]f32{ 0x1p-20, 0x1p100 };
    var nv: [2 * 18 * 2]u8 = undefined;
    var want: [2 * 18]u8 = undefined;
    for (0..2) |x| for (pow2, 0..) |b, j| {
        nv[(x * 18 + j) * 2] = b;
        nv[(x * 18 + j) * 2 + 1] = b;
        want[x * 18 + j] = converterCode(.{ b, b }, gl[x]) orelse return error.ConverterRefused;
    };
    {
        const q = u8Array(&nv, &.{ 2, 1, 36 });
        defer _ = mlx.mlx_array_free(q);
        const g = try bf16Array(&gl);
        defer _ = mlx.mlx_array_free(g);
        const v = try scales(gpa, s, q, g);
        if (v != .mx) return error.Refused;
        defer _ = mlx.mlx_array_free(v.mx);
        const got = try bytesOf(v.mx);
        defer gpa.free(got);
        try testing.expectEqualSlices(u8, &want, got);
        try testing.expectEqualSlices(c_int, &.{ 2, 1, 18 }, mlx.getShape(v.mx));
    }
    // One planted span beside a plain one (2^0 under 2^-10): the view's verdict against the converter's.
    const Case = struct { pair: [2]u8, global: f32, view: bool };
    const cases = [_]Case{
        .{ .pair = .{ 0x01, 0x01 }, .global = 0x1p-117, .view = true }, // code 1
        .{ .pair = .{ 0x78, 0x78 }, .global = 0x1p119, .view = true }, // code 254
        .{ .pair = .{ 0x01, 0x01 }, .global = 0x1p-118, .view = false }, // code 0: the converter writes it, MLX reads a subnormal
        .{ .pair = .{ 0x78, 0x78 }, .global = 0x1p120, .view = false }, // past code 254
        .{ .pair = .{ 0x38, 0x40 }, .global = 0x1p-10, .view = false }, // two powers of two
        .{ .pair = .{ 0x39, 0x39 }, .global = 0x1p-10, .view = false }, // 1.125
        .{ .pair = .{ 0x03, 0x03 }, .global = 0x1p-10, .view = false }, // the subnormal 3 x 2^-9
        .{ .pair = .{ 0x00, 0x00 }, .global = 0x1p-10, .view = false }, // zero
        .{ .pair = .{ 0x80, 0x80 }, .global = 0x1p-10, .view = false }, // -0
        .{ .pair = .{ 0xB8, 0xB8 }, .global = 0x1p-10, .view = false }, // -1
        .{ .pair = .{ 0x7F, 0x7F }, .global = 0x1p-10, .view = false }, // NaN
    };
    for (cases) |c| {
        // Expert 0 holds the planted span and a 2^0 pair; expert 1 two 2^0 pairs under 2^-10.
        const bytes = [_]u8{ c.pair[0], c.pair[1], 0x38, 0x38, 0x38, 0x38, 0x38, 0x38 };
        const gl2 = [2]f32{ c.global, 0x1p-10 };
        const conv = converterCode(c.pair, c.global);
        const q = u8Array(&bytes, &.{ 2, 1, 4 });
        defer _ = mlx.mlx_array_free(q);
        const g = try bf16Array(&gl2);
        defer _ = mlx.mlx_array_free(g);
        const v = try scales(gpa, s, q, g);
        defer if (v == .mx) {
            _ = mlx.mlx_array_free(v.mx);
        };
        std.debug.print("[mxfp4 view] pair 0x{x:0>2} 0x{x:0>2} under {e}: view {s}, converter code {d} (-1: refused)\n", .{ c.pair[0], c.pair[1], c.global, @tagName(v), if (conv) |x| @as(i32, x) else -1 });
        try testing.expectEqual(c.view, v == .mx);
        // The converter takes every span the view takes, at the same code; it parts only at code 0.
        if (c.view) {
            const got = try bytesOf(v.mx);
            defer gpa.free(got);
            try testing.expectEqual(conv.?, got[0]);
            try testing.expectEqual(converterCode(.{ 0x38, 0x38 }, 0x1p-10).?, got[2]);
        } else {
            try testing.expectEqual(@as(u64, 1), v.nv.spans);
            try testing.expectEqual(c.global == 0x1p-118, conv != null);
        }
    }
    // A global off a power of two keeps the bank whole, named by its expert.
    {
        const bytes: [8]u8 = @splat(0x38);
        const q = u8Array(&bytes, &.{ 2, 1, 4 });
        defer _ = mlx.mlx_array_free(q);
        const g = try bf16Array(&.{ 0x1p-10, 1.5 * 0x1p-10 });
        defer _ = mlx.mlx_array_free(g);
        const v = try scales(gpa, s, q, g);
        try testing.expect(v == .nv and v.nv == .global and v.nv.global.expert == 1);
        try testing.expectEqual(@as(?u8, null), converterCode(.{ 0x38, 0x38 }, 1.5 * 0x1p-10));
    }
}

// ── the view through the plugin's load ──

/// A copy of `w` holding the same arrays: each load takes its map's entries.
fn cloneMap(w: *const resident.Weights) !resident.Weights {
    var out = resident.Weights.init(testing.allocator);
    errdefer out.deinit();
    var it = w.map.iterator();
    while (it.next()) |kv| {
        var a = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&a, kv.value_ptr.*);
        out.put(kv.key_ptr.*, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
    }
    return out;
}

/// Trunk layers [lo, hi) at the release's shapes with `E` routed experts in mxfp4 (`Fixture.putLayer`): only the map.
fn mxLayers(cfg: *const Config, lo: usize, hi: usize, E: u32) !*fixture.Fixture {
    const fx = try testing.allocator.create(fixture.Fixture);
    fx.* = .{ .gpa = testing.allocator, .m = undefined, .w = resident.Weights.init(testing.allocator), .seed = 0x3E57 };
    errdefer {
        fx.w.deinit();
        testing.allocator.destroy(fx);
    }
    for (lo..hi) |li| try fx.putLayer(cfg, "layers.{d}.", li, E, true, null);
    return fx;
}

fn dropMap(fx: *fixture.Fixture) void {
    fx.w.deinit();
    testing.allocator.destroy(fx);
}

const projs = [_][]const u8{ "w1", "w3", "w2" };
const global_names = [_][]const u8{ "gate_proj.expert_scale", "up_proj.expert_scale", "expert_scale" };

fn expertName(buf: []u8, li: usize, x: usize, p: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.ffn.experts.{d}.{s}.scales", .{ li, x, projs[p] });
}

fn globalName(buf: []u8, li: usize, p: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.ffn.experts.{s}", .{ li, global_names[p] });
}

/// Layer `li`'s mxfp4 experts in `w` as NVIDIA's NVFP4 pack stores the same values: the codes kept, each 32-span's
/// e8m0 2^k as two e4m3 2^(k - b) under the expert's global 2^b (`[E]` bf16 under the converter's names). Expert 0's b
/// puts its least scale at e4m3's least power of two (a subnormal), expert 1's its greatest at 2^8.
fn toNv(w: *resident.Weights, li: usize, E: usize, rng: std.Random) !void {
    const gpa = testing.allocator;
    var gl: [3][64]f32 = undefined;
    for (0..3) |p| for (0..E) |x| {
        var buf: [160]u8 = undefined;
        const name = try expertName(&buf, li, x, p);
        const old = w.get(name) orelse return error.MissingWeight;
        const rows = mlx.getShape(old)[0];
        const cols = mlx.getShape(old)[1];
        const mx = try bytesOf(old);
        defer gpa.free(mx);
        var lo: i32 = 127;
        var hi: i32 = -127;
        for (mx) |c| {
            lo = @min(lo, @as(i32, c) - 127);
            hi = @max(hi, @as(i32, c) - 127);
        }
        if (hi - 8 > lo + 9) return error.ScaleSpread;
        const b: i32 = if (x == 0) lo + 9 else if (x == 1) hi - 8 else rng.intRangeAtMost(i32, hi - 8, lo + 9);
        const nv = try gpa.alloc(u8, 2 * mx.len);
        defer gpa.free(nv);
        for (mx, 0..) |c, i| {
            nv[2 * i] = e4m3Of(@as(i32, c) - 127 - b);
            nv[2 * i + 1] = nv[2 * i];
        }
        try w.put(name, u8Array(nv, &.{ rows, 2 * cols }));
        gl[p][x] = std.math.ldexp(@as(f32, 1), b);
    };
    for (0..3) |p| {
        var buf: [160]u8 = undefined;
        try w.put(try globalName(&buf, li, p), try bf16Array(gl[p][0..E]));
    }
}

/// `f` applied to the e4m3 bytes (rows of `cols`) of expert `x`'s `p` scales in `w`.
fn editScales(w: *resident.Weights, li: usize, x: usize, p: usize, f: *const fn ([]u8, usize) void) !void {
    var buf: [160]u8 = undefined;
    const name = try expertName(&buf, li, x, p);
    const old = w.get(name) orelse return error.MissingWeight;
    const rows = mlx.getShape(old)[0];
    const cols = mlx.getShape(old)[1];
    const b = try bytesOf(old);
    defer testing.allocator.free(b);
    f(b, @intCast(cols));
    try w.put(name, u8Array(b, &.{ rows, cols }));
}

/// The first span's two scales two powers of two apart: no e8m0 code holds the span.
fn splitPair(b: []u8, _: usize) void {
    const a: i32 = e4m3Exponent(b[0]).?;
    b[1] = e4m3Of(if (a < 8) a + 1 else a - 1);
}

/// Every span one power of two lower (one higher at the least): still a view, of other values.
fn shiftAll(b: []u8, _: usize) void {
    for (b) |*v| {
        const a: i32 = e4m3Exponent(v.*).?;
        v.* = e4m3Of(if (a > -9) a - 1 else a + 1);
    }
}

/// Expert `x`'s `p` global times 1.5: no power of two.
fn offGlobal(w: *resident.Weights, li: usize, x: usize, p: usize) !void {
    var buf: [160]u8 = undefined;
    const name = try globalName(&buf, li, p);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, w.get(name).?, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    var v: [64]f32 = undefined;
    const n = mlx.mlx_array_size(f);
    @memcpy(v[0..n], (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..n]);
    v[x] *= 1.5;
    try w.put(name, try bf16Array(v[0..n]));
}

/// The plugin's load of `w` as a pair worker holding [lo, hi) (the decode fixture's), the view on or off.
fn load(cfg: *const Config, w: resident.Weights, lo: usize, hi: usize, view: bool) !*fixture.Fixture {
    const gpa = testing.allocator;
    const fx = try gpa.create(fixture.Fixture);
    fx.* = .{ .gpa = gpa, .m = undefined, .w = w, .seed = 0x51EE };
    errdefer {
        fx.w.deinit();
        gpa.destroy(fx);
    }
    const sw = env.override(.MLX_SERVE_DSV41_MXFP4_VIEW, if (view) "1" else null);
    defer sw.restore();
    fx.m = try dsv41.init(gpa, cfg, &fx.w, mlx.gpuStream(), .{ .pair = .{ .lo = lo, .hi = hi, .worker = true }, .act = .bfloat16 });
    errdefer fx.m.deinit();
    fx.m.dec_state = try dsv41.initState(fx.m, gpa);
    return fx;
}

/// Values whose bits differ between `a` and `b` (one dtype and shape), printed under `label`.
fn bitDiff(label: []const u8, a: A, b: A) !usize {
    const x = try bytesOf(a);
    defer testing.allocator.free(x);
    const y = try bytesOf(b);
    defer testing.allocator.free(y);
    if (x.len != y.len or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return error.ShapeMismatch;
    const n = mlx.mlx_array_itemsize(a);
    var diff: usize = 0;
    var i: usize = 0;
    while (i < x.len) : (i += n) diff += @intFromBool(!std.mem.eql(u8, x[i..][0..n], y[i..][0..n]));
    std.debug.print("[mxfp4 view] {s}: {d} of {d} values differ\n", .{ label, diff, x.len / n });
    return diff;
}

/// The relative rms error of `a` against `b` in f32, `a`'s bits that differ printed beside it under `label`.
fn relErr(label: []const u8, a: A, b: A) !f32 {
    _ = try bitDiff(label, a, b);
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const y = try sc.astype(b, .float32);
    const d = try sc.op2(mlx.mlx_subtract, try sc.astype(a, .float32), y);
    var sums: [2]f32 = undefined;
    for ([_]A{ d, y }, &sums) |x, *o| {
        var n = mlx.mlx_array_new();
        const t = try sc.res(mlx.mlx_sum(&n, try sc.mul(x, x), false, sc.s), &n);
        try mlx.check(mlx.mlx_array_eval(t));
        try mlx.check(mlx.mlx_array_item_float32(o, t));
    }
    const e = @sqrt(sums[0] / sums[1]);
    std.debug.print("[mxfp4 view] {s}: relative rms error {e}\n", .{ label, e });
    return e;
}

/// A reordered sum (16-groups summed apart, other tiling) moves bf16 outputs by an ulp here and there, far below this;
/// a wrong scale moves them by tens of percent.
const tol: f32 = 1e-2;

fn sameBytes(a: A, b: A) !bool {
    const x = try bytesOf(a);
    defer testing.allocator.free(x);
    const y = try bytesOf(b);
    defer testing.allocator.free(y);
    return std.mem.eql(u8, x, y) and std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b));
}

fn isFp4(b: dsv41.Bank, mode: QuantMode, group: u32) bool {
    return b.qp.bits == 4 and b.qp.mode == mode and b.qp.group_size == group;
}

/// `bank`'s values (`[E, out, in]` f32) as MLX dequantizes them, each expert's rows [lo, hi) times its global in `g`.
fn valuesOf(sc: *dsv41.Scope, b: dsv41.Bank, g: []const ?A, halves: usize) !A {
    var d = mlx.mlx_array_new();
    const v = try sc.res(mlx.mlx_dequantize(&d, b.w, b.s, none, mlx.mlx_optional_int.some(@intCast(b.qp.group_size)), mlx.mlx_optional_int.some(4), b.qp.mode.cstr(), none, .{ .value = .float32, .has_value = true }, sc.s), &d);
    if (g[0] == null) return v;
    const rows = @as(usize, @intCast(mlx.getShape(v)[1])) / halves;
    var parts: [2]A = undefined;
    for (parts[0..halves], 0..) |*p, i| p.* = try sc.mul(try sc.slice(v, 1, i * rows, (i + 1) * rows), try sc.reshape(try sc.astype(g[i].?, .float32), &.{ -1, 1, 1 }));
    return sc.concat(parts[0..halves], 1);
}

/// Every value of `a` and `b` (f32) bit for bit.
fn sameValues(sc: *dsv41.Scope, a: A, b: A) !bool {
    var o = mlx.mlx_array_new();
    const eq = try sc.res(mlx.mlx_array_equal(&o, try sc.view(a, .uint32), try sc.view(b, .uint32), false, sc.s), &o);
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

test "dsv41 mxfp4 view: NVIDIA's power-of-two NVFP4 bank loads as the MXFP4 bank of its values (the codes as stored, the converter's e8m0, no e4m3 scales or globals) and serves the NVFP4 bank's outputs to a reorder's tolerance: decode R = 1..6 (fused fp4, the gathers), prefill, the whole layer; an MXFP4 bank loads as it is" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fp4_nax = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer fp4_nax.restore();
    const li = 20;
    const E = 16;
    var cfg = try fixture.releaseConfig();
    cfg.num_experts = E;
    const base = try mxLayers(&cfg, li, li + 1, E);
    defer dropMap(base);
    var nv_map = try cloneMap(&base.w);
    defer nv_map.deinit();
    var prng = std.Random.DefaultPrng.init(0x3E57);
    try toNv(&nv_map, li, E, prng.random());

    const nv = try load(&cfg, try cloneMap(&nv_map), li, li + 1, false);
    defer nv.deinit();
    const view = try load(&cfg, try cloneMap(&nv_map), li, li + 1, true);
    defer view.deinit();
    const mx = try load(&cfg, try cloneMap(&base.w), li, li + 1, false);
    defer mx.deinit();
    const mx_on = try load(&cfg, try cloneMap(&base.w), li, li + 1, true);
    defer mx_on.deinit();
    // The planted control: every expert's gate one power of two off, still a view.
    const off = blk: {
        var w = try cloneMap(&nv_map);
        for (0..E) |x| editScales(&w, li, x, 0, &shiftAll) catch |e| {
            w.deinit();
            return e;
        };
        break :blk try load(&cfg, w, li, li + 1, true);
    };
    defer off.deinit();

    // The banks: the view's are the MXFP4 pack's, byte for byte, and the NVFP4 load's codes.
    const ev = &view.layer(li).experts;
    const en = &nv.layer(li).experts;
    const em = &mx.layer(li).experts;
    const eo = &mx_on.layer(li).experts;
    try testing.expect(isFp4(en.gu.?, .nvfp4, 16) and isFp4(en.down, .nvfp4, 16) and en.scale[0] != null);
    for ([_]*const dsv41.Experts{ ev, em, eo }) |e| {
        try testing.expect(isFp4(e.gu.?, .mxfp4, 32) and isFp4(e.down, .mxfp4, 32));
        for (e.scale) |g| try testing.expect(g == null);
    }
    for ([_]*const dsv41.Experts{ ev, eo }) |e| {
        try testing.expect(try sameBytes(e.gu.?.w, em.gu.?.w) and try sameBytes(e.gu.?.s, em.gu.?.s));
        try testing.expect(try sameBytes(e.down.w, em.down.w) and try sameBytes(e.down.s, em.down.s));
    }
    try testing.expect(try sameBytes(ev.gu.?.w, en.gu.?.w) and try sameBytes(ev.down.w, en.down.w));
    var nb: [160]u8 = undefined;
    for (0..3) |p| {
        try testing.expect(view.w.get(try globalName(&nb, li, p)) == null);
        try testing.expect(nv.w.get(try globalName(&nb, li, p)) != null);
    }
    // Its tally: each scale byte pair of the three banks one byte, the globals gone.
    const t = view.m.view;
    const d = cfg.hidden_size;
    const n = cfg.moe_intermediate_size;
    try testing.expect(t.viewed == 1 and t.refused == 0 and mx_on.m.view.viewed == 0);
    try testing.expectEqual(@as(u64, E) * 3 * d * n / 32, t.mx_bytes);
    try testing.expectEqual(2 * t.mx_bytes + 3 * E * 2, t.nv_bytes);
    // The converter's --check (`mx_same`): every value of the NVFP4 bank times its globals is the view's.
    {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        try testing.expect(try sameValues(&sc, try valuesOf(&sc, en.gu.?, en.scale[0..2], 2), try valuesOf(&sc, ev.gu.?, &.{null}, 1)));
        try testing.expect(try sameValues(&sc, try valuesOf(&sc, en.down, en.scale[2..3], 1), try valuesOf(&sc, ev.down, &.{null}, 1)));
    }

    const MR = decode.max_rows;
    const hc = cfg.dsv4_hc_mult;
    const arms = [_]*fixture.Fixture{ view, mx };
    var worst: f32 = 0;
    var control: f32 = std.math.inf(f32);
    var label: [80]u8 = undefined;
    // Decode: the fused fp4 kernels (they must take the layer), then the routed gathers.
    for (1..MR + 1) |R| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h = try sc.keep(try nv.normal(&.{ ci(R), ci(d) }, .bfloat16, 0, 1));
        const x1 = try sc.keep(try nv.stream(R));
        const sp: dsv41.Split = .{ .pre = none, .post = try sc.keep(try nv.uniform(&.{ ci(R), ci(hc) }, 0.05, 1.5)), .comb = try sc.keep(try nv.uniform(&.{ ci(R), ci(hc), ci(hc) }, -0.5, 0.5)) };
        var fused: [4]A = undefined;
        var gathers: [4]A = undefined;
        for ([_]*fixture.Fixture{ nv, view, mx, off }, &fused, &gathers) |fx, *y, *z| {
            const ly = fx.layer(li);
            const r = try dsv41.route(&sc, fx.m, ly, h, none);
            y.* = (try dsv41.ffnDecode(&sc, fx.m, ly, h, r.idx, r.wts, x1, &sp)) orelse return error.Fp4KernelsDeclined;
            z.* = try dsv41.routed(&sc, fx.m, ly, h, r.idx, r.wts);
        }
        for (arms, 1..) |_, k| {
            worst = @max(worst, try relErr(try std.fmt.bufPrint(&label, "decode fused R={d}, {s}", .{ R, if (k == 1) "view" else "mxfp4 pack" }), fused[k], fused[0]));
            worst = @max(worst, try relErr(try std.fmt.bufPrint(&label, "decode gathers R={d}, {s}", .{ R, if (k == 1) "view" else "mxfp4 pack" }), gathers[k], gathers[0]));
        }
        control = @min(control, try relErr(try std.fmt.bufPrint(&label, "decode fused R={d}, planted control", .{R}), fused[3], fused[0]));
        control = @min(control, try relErr(try std.fmt.bufPrint(&label, "decode gathers R={d}, planted control", .{R}), gathers[3], gathers[0]));
    }
    // Prefill: the sorted gathers.
    for ([_]usize{ 512, 2048 }) |L| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h = try sc.keep(try nv.normal(&.{ ci(L), ci(d) }, .bfloat16, 0, 1));
        var y: [4]A = undefined;
        for ([_]*fixture.Fixture{ nv, view, mx, off }, &y) |fx, *o| {
            const r = try dsv41.route(&sc, fx.m, fx.layer(li), h, none);
            o.* = try sc.keep(try dsv41.moeShare(fx.m, gpa, li, h, r.idx, r.wts));
        }
        for (arms, 1..) |_, k| worst = @max(worst, try relErr(try std.fmt.bufPrint(&label, "prefill L={d}, {s}", .{ L, if (k == 1) "view" else "mxfp4 pack" }), y[k], y[0]));
        control = @min(control, try relErr(try std.fmt.bufPrint(&label, "prefill L={d}, planted control", .{L}), y[3], y[0]));
    }
    // The whole layer: a 512-row prompt, then steps of 1..6 rows on its state (the stream and the mix).
    var p0: usize = 0;
    for ([_]usize{ 512, 1, 2, 3, 4, 5, 6 }) |L| {
        const x = try nv.stream(L);
        defer _ = mlx.mlx_array_free(x);
        const pre = try nv.cpre(L);
        defer _ = mlx.mlx_array_free(pre);
        var out: [3][2]A = undefined;
        var have: usize = 0;
        defer for (out[0..have]) |o| for (o) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]*fixture.Fixture{ nv, view, mx }) |fx| {
            out[have] = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
            have += 1;
        }
        for (1..3) |k| for (0..2) |j| {
            worst = @max(worst, try relErr(try std.fmt.bufPrint(&label, "layer L={d} at {d} {s}, {s}", .{ L, p0, if (j == 0) "stream" else "mix", if (k == 1) "view" else "mxfp4 pack" }), out[k][j], out[0][j]));
        };
        p0 += L;
    }
    try testing.expect(worst <= tol);
    // A view one power of two off on every gate span: the comparison sees a wrong scale.
    try testing.expect(control > 10 * tol);
}

test "dsv41 mxfp4 view: a bank with one span or one global off a power of two stays NVFP4 as loaded, named in the log, beside a layer the view takes; each layer's MoE serves the NVFP4 load's outputs at the defaults to a reorder's tolerance, and swapped gate and up globals fail" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const fp4_nax = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer fp4_nax.restore();
    const lo = 20;
    const hi = 23;
    const E = 16;
    var cfg = try fixture.releaseConfig();
    cfg.num_experts = E;
    const base = try mxLayers(&cfg, lo, hi, E);
    defer dropMap(base);
    var nv_map = try cloneMap(&base.w);
    defer nv_map.deinit();
    var prng = std.Random.DefaultPrng.init(0x0FF);
    for (lo..hi) |li| try toNv(&nv_map, li, E, prng.random());
    try editScales(&nv_map, 21, 3, 2, &splitPair);
    try offGlobal(&nv_map, 22, 5, 1);

    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var root: [512]u8 = undefined;
    var pb: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&pb, "{s}/view.log", .{root[0..try td.dir.realPath(io, &root)]});
    try log.openFile(path, 0);
    const view = load(&cfg, try cloneMap(&nv_map), lo, hi, true) catch |e| {
        log.closeFile();
        return e;
    };
    log.closeFile();
    defer view.deinit();
    const nv = try load(&cfg, try cloneMap(&nv_map), lo, hi, false);
    defer nv.deinit();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "layers.21.ffn.experts stays NVFP4") != null);
    try testing.expect(std.mem.indexOf(u8, text, "layers.22.ffn.experts stays NVFP4") != null);
    try testing.expect(std.mem.indexOf(u8, text, "layers.20.ffn.experts stays") == null);
    try testing.expect(view.m.view.viewed == 1 and view.m.view.refused == 2);
    try testing.expect(isFp4(view.layer(20).experts.gu.?, .mxfp4, 32) and view.layer(20).experts.scale[0] == null);
    var nb: [160]u8 = undefined;
    for ([_]usize{ 21, 22 }) |li| {
        const a = &view.layer(li).experts;
        const b = &nv.layer(li).experts;
        try testing.expect(isFp4(a.gu.?, .nvfp4, 16) and isFp4(a.down, .nvfp4, 16));
        try testing.expect(try sameBytes(a.gu.?.w, b.gu.?.w) and try sameBytes(a.gu.?.s, b.gu.?.s) and try sameBytes(a.down.w, b.down.w) and try sameBytes(a.down.s, b.down.s));
        for (a.scale, b.scale, 0..) |x, y, p| {
            try testing.expect(try sameBytes(x.?, y.?));
            try testing.expect(view.w.get(try globalName(&nb, li, p)) != null);
        }
    }
    // Each layer alone on one input, the view's mixed formats against the NVFP4 load at the defaults (MOE_PAIR_FP4
    // on): the prompt's MoE and decode's at 1 and 2 rows. One layer's routes are the same in both, so its reordered
    // scale sums stay at their own size; chained, top-k picks amplify them.
    var worst: f32 = 0;
    var label: [64]u8 = undefined;
    for (lo..hi) |li| worst = @max(worst, try layerErr(&label, view, nv, li));
    try testing.expect(worst <= tol);
    // Planted: the NVFP4 load's layer 20 with its gate and up globals swapped.
    swapGlobals(&nv.layer(20).experts);
    defer swapGlobals(&nv.layer(20).experts);
    try testing.expect(try layerErr(&label, view, nv, 20) > 10 * tol);
}

fn swapGlobals(e: *dsv41.Experts) void {
    e.scale = .{ e.scale[1], e.scale[0], e.scale[2] };
}

/// The worst relative rms error of layer `li`'s MoE in `a` against `b`: a 256-row prompt, then decode at 1 and 2 rows.
fn layerErr(label: []u8, a: *fixture.Fixture, b: *fixture.Fixture, li: usize) !f32 {
    const gpa = testing.allocator;
    const d = b.m.cfg.hidden_size;
    const hc = b.m.cfg.dsv4_hc_mult;
    var worst: f32 = 0;
    for ([_]usize{ 256, 1, 2 }) |L| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h = try sc.keep(try b.normal(&.{ ci(L), ci(d) }, .bfloat16, 0, 1));
        const x1 = try sc.keep(try b.stream(L));
        const sp: dsv41.Split = .{ .pre = none, .post = try sc.keep(try b.uniform(&.{ ci(L), ci(hc) }, 0.05, 1.5)), .comb = try sc.keep(try b.uniform(&.{ ci(L), ci(hc), ci(hc) }, -0.5, 0.5)) };
        var y: [2]A = undefined;
        for ([_]*fixture.Fixture{ a, b }, &y) |fx, *o| {
            const ly = fx.layer(li);
            const r = try dsv41.route(&sc, fx.m, ly, h, none);
            o.* = if (L > decode.max_rows)
                try sc.keep(try dsv41.moeShare(fx.m, gpa, li, h, r.idx, r.wts))
            else
                (try dsv41.ffnDecode(&sc, fx.m, ly, h, r.idx, r.wts, x1, &sp)) orelse return error.Fp4KernelsDeclined;
        }
        worst = @max(worst, try relErr(try std.fmt.bufPrint(label, "layer {d} L={d}", .{ li, L }), y[0], y[1]));
    }
    return worst;
}
