//! sdk/testing/parity.
//! Owns hostF32, rmsVsTruth, testReadF32, jsonF64, jsonToMlxArray, maxAbsDiffF32, decidedMisses, and 12 more.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const Weights = @import("../weights/resident.zig").Weights;

/// An evaluated f32 array's host values (test inputs only: reading migrates the buffer).
pub fn hostF32(a: mlx.mlx_array) ![]const f32 {
    if (mlx.mlx_array_dtype(a) != .float32) return error.NotFloat32;
    try mlx.check(mlx.mlx_array_eval(a));
    const d = mlx.mlx_array_data_float32(a) orelse return error.NoData;
    return d[0..mlx.mlx_array_size(a)];
}

/// Test helper: eval `arr`, flatten, and copy the first `out.len` f32 values to host.
pub fn testReadF32(arr: mlx.mlx_array, out: []f32, s: mlx.mlx_stream) !void {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const sh = [_]c_int{@intCast(out.len)};
    try mlx.check(mlx.mlx_reshape(&flat, f, &sh, 1, s));
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    _ = mlx.mlx_vector_array_append_value(ev, flat);
    try mlx.check(mlx.mlx_eval(ev));
    const data = mlx.mlx_array_data_float32(flat) orelse return error.TestUnexpectedNullData;
    @memcpy(out, data[0..out.len]);
}

pub fn jsonF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

/// Max |a-b| between two same-shaped float arrays, with a NaN guard (a NaN
/// candidate must FAIL, never score 0 — the parity-loop finiteness rule).
pub fn maxAbsDiffF32(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    // materializedOwnedCopy: a same-dtype astype is a NO-OP view and a slice's
    // raw data pointer walks the PARENT's physical layout — force a contiguous
    // owned buffer before touching bytes.
    var a_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_t);
    try mlx.check(mlx.mlx_astype(&a_t, a, .float32, s));
    const a32 = try materializedOwnedCopy(s, a_t);
    defer _ = mlx.mlx_array_free(a32);
    var b_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_t);
    try mlx.check(mlx.mlx_astype(&b_t, b, .float32, s));
    const b32 = try materializedOwnedCopy(s, b_t);
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    const pa = mlx.mlx_array_data_float32(a32) orelse return error.NoData;
    const pb = mlx.mlx_array_data_float32(b32) orelse return error.NoData;
    var max_err: f32 = 0;
    for (0..n) |i| {
        const d = @abs(pa[i] - pb[i]);
        if (!std.math.isFinite(d)) return error.NonFiniteValue;
        if (d > max_err) max_err = d;
    }
    return max_err;
}

pub fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

/// A bf16 forward against an fp32 reference carries up to ~0.5 of logit error on a tiny model, so only a row whose
/// reference top-2 margin clears that decides its argmax.
const DECIDED_MARGIN: f32 = 0.5;

/// Rows whose argmax differs from the reference where its margin decides it;
/// near-tied flips are printed, never counted.
pub fn decidedMisses(ours: []const f32, ref: []const f32, rows: usize, v: usize) usize {
    var misses: usize = 0;
    for (0..rows) |i| {
        const o = ours[i * v ..][0..v];
        const r = ref[i * v ..][0..v];
        const io_ = std.mem.indexOfMax(f32, o);
        const ir = std.mem.indexOfMax(f32, r);
        if (io_ == ir) continue;
        var second: f32 = -std.math.inf(f32);
        for (r, 0..) |x, j| {
            if (j != ir) second = @max(second, x);
        }
        const gap = r[ir] - second;
        if (gap >= DECIDED_MARGIN) misses += 1;
        std.debug.print("[fixture] row {d}: argmax {d} vs reference {d} (reference margin {d:.3}{s})\n", .{ i, io_, ir, gap, if (gap >= DECIDED_MARGIN) "" else ", a near tie" });
    }
    return misses;
}

pub fn qsaArraysAllEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, a, b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var out: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&out, all));
    return out;
}

// ── tests ──


/// Root-mean-square error of an array against an f64 truth of the same size.
pub fn rmsVsTruth(a: mlx.mlx_array, truth: []const f64) !f64 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, mlx.gpuStream()));
    const d = try hostF32(f);
    if (d.len != truth.len) return error.SizeMismatch;
    var e: f64 = 0;
    for (truth, d) |t, x| e += (@as(f64, x) - t) * (@as(f64, x) - t);
    return @sqrt(e / @as(f64, @floatFromInt(truth.len)));
}

pub fn jsonToMlxArray(allocator: std.mem.Allocator, v: std.json.Value) !mlx.mlx_array {
    var shape: [8]c_int = undefined;
    var ndim: usize = 0;
    var cur = v;
    while (cur == .array) {
        if (ndim >= 8) return error.TooManyDims;
        shape[ndim] = @intCast(cur.array.items.len);
        ndim += 1;
        if (cur.array.items.len == 0) break;
        cur = cur.array.items[0];
    }
    var total: usize = 1;
    for (shape[0..ndim]) |d| total *= @intCast(d);
    const data = try allocator.alloc(f32, total);
    defer allocator.free(data);
    var idx: usize = 0;
    const Fill = struct {
        fn fill(val: std.json.Value, out: []f32, i: *usize) void {
            switch (val) {
                .array => |a| for (a.items) |item| fill(item, out, i),
                else => {
                    out[i.*] = @floatCast(jsonF64(val));
                    i.* += 1;
                },
            }
        }
    };
    Fill.fill(v, data, &idx);
    return mlx.mlx_array_new_data(data.ptr, &shape, @intCast(ndim), .float32);
}

/// Eval an mlx array as f32 and copy its data into a fresh slice.
pub fn evalToF32(al: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.InvalidDtype;
    const out = try al.alloc(f32, n);
    @memcpy(out, d[0..n]);
    return out;
}

/// Uniform [lo, hi) drawn by MLX: a host fill of a vocab-sized weight in a
/// Debug test build dominated the whole suite's runtime.
pub fn testRandUniformBf16(rnd: std.Random, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

pub fn testRandWeightBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return testRandUniformBf16(rnd, shape, -0.5, 0.5, s);
}

pub fn qwen4ReadF32(alloc: std.mem.Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const out = try alloc.alloc(f32, n);
    const src = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    @memcpy(out, src[0..n]);
    return out;
}

pub const Qwen4RowCmp = struct { min_cos: f64, argmax_agree: usize, rows: usize };

/// Per-row cosine + argmax agreement between `a` and `b`, both `[rows][v]`.
/// The tiny random oracle's near ties, acquitted by the REFERENCE's own
/// numbers, never ours: a row whose QSA block selection (`qsa_gap`, the
/// relative margin between the last selected and first rejected block score,
/// min over the QSA layers — relu leaves exact zeros) sits under 3% flips
/// under 1% kernel noise and its whole row differs; a row whose top-2 logit
/// margin (`logit_margin`) is under 3% of the row's RMS logit has a coin-flip
/// argmax. Rows without a tensor are never acquitted.
const QWEN4_TIE_REL: f32 = 0.03;
/// Expert-selection acquittal: the router sees a bf16 residual (noisier than
/// the f32 QSA scores) — a k=2 tiny fixture flipped a row at a 3.5% margin.
const QWEN4_TIE_REL_ROUTE: f32 = 0.05;

pub const Qwen4Ties = struct {
    qsa_gap: ?[]const f32,
    /// Min over MoE layers of the relative softmax margin at the top-k
    /// boundary (k < E fixtures; absent or 1.0 when every expert runs).
    route_gap: ?[]const f32,
    logit_margin: ?[]const f32,
    pub fn read(alloc: std.mem.Allocator, fx: *const Weights, s: mlx.mlx_stream) !Qwen4Ties {
        return .{
            .qsa_gap = if (fx.get("qsa_gap")) |g| try qwen4ReadF32(alloc, g, s) else null,
            .route_gap = if (fx.get("route_gap")) |g| try qwen4ReadF32(alloc, g, s) else null,
            .logit_margin = if (fx.get("logit_margin")) |g| try qwen4ReadF32(alloc, g, s) else null,
        };
    }
    pub fn deinit(self: Qwen4Ties, alloc: std.mem.Allocator) void {
        if (self.qsa_gap) |g| alloc.free(g);
        if (self.route_gap) |g| alloc.free(g);
        if (self.logit_margin) |g| alloc.free(g);
    }
    /// Row-level acquittal: a QSA block selection OR an expert selection
    /// near-tie in the reference.
    pub fn rowTied(self: Qwen4Ties, row: usize) bool {
        if (self.qsa_gap) |g| if (g[row] < QWEN4_TIE_REL) return true;
        if (self.route_gap) |g| if (g[row] < QWEN4_TIE_REL_ROUTE) return true;
        return false;
    }
    pub fn margin(self: Qwen4Ties, row: usize) f32 {
        var m: f32 = 1.0;
        if (self.qsa_gap) |g| m = @min(m, g[row]);
        if (self.route_gap) |g| m = @min(m, g[row]);
        return m;
    }
    /// Argmax acquittal: row-tied OR a near-tie top-2 logit.
    pub fn argmaxTied(self: Qwen4Ties, row: usize, ref_row: []const f32) bool {
        if (self.rowTied(row)) return true;
        const mg = self.logit_margin orelse return false;
        var ss: f64 = 0;
        for (ref_row) |x| ss += @as(f64, x) * x;
        const rms: f32 = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(ref_row.len))));
        return mg[row] < QWEN4_TIE_REL * rms;
    }
};

pub const Qwen4TiedCmp = struct { min_cos: f64, agree: usize, decisive: usize, tied: usize };

/// Row-wise compare over `rows` starting at fixture row `row0`: min cos over
/// non-tied rows, argmax agreement over decisive rows; prints every acquittal.
pub fn qwen4CompareTied(tag: []const u8, a: []const f32, b: []const f32, rows: usize, v: usize, row0: usize, ties: Qwen4Ties) Qwen4TiedCmp {
    var out: Qwen4TiedCmp = .{ .min_cos = 1.0, .agree = 0, .decisive = 0, .tied = 0 };
    for (0..rows) |r| {
        const ref_row = b[r * v ..][0..v];
        const rc = qwen4CompareRows(a[r * v ..][0..v], ref_row, 1, v);
        if (ties.rowTied(row0 + r)) {
            out.tied += 1;
            std.debug.print("    {s} row {d}: acquitted (reference selection margin {d:.4}), cos {d:.4}\n", .{ tag, row0 + r, ties.margin(row0 + r), rc.min_cos });
            continue;
        }
        if (rc.min_cos < out.min_cos) out.min_cos = rc.min_cos;
        if (env.on(.QWEN4_FIXTURE_ROWS)) std.debug.print("    {s} row {d}: cos {d:.4} margin {d:.4}\n", .{ tag, row0 + r, rc.min_cos, ties.margin(row0 + r) });
        if (ties.argmaxTied(row0 + r, ref_row)) {
            std.debug.print("    {s} row {d}: argmax acquitted (reference top-2 margin {d:.4})\n", .{ tag, row0 + r, ties.logit_margin.?[row0 + r] });
            continue;
        }
        out.decisive += 1;
        out.agree += rc.argmax_agree;
    }
    return out;
}

pub fn qwen4CompareRows(a: []const f32, b: []const f32, rows: usize, v: usize) Qwen4RowCmp {
    var r: Qwen4RowCmp = .{ .min_cos = 1.0, .argmax_agree = 0, .rows = rows };
    for (0..rows) |i| {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var ia: usize = 0;
        var ib: usize = 0;
        for (0..v) |j| {
            const x: f64 = a[i * v + j];
            const y: f64 = b[i * v + j];
            dot += x * y;
            na += x * x;
            nb += y * y;
            if (x > a[i * v + ia]) ia = j;
            if (y > b[i * v + ib]) ib = j;
        }
        const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
        if (cos < r.min_cos) r.min_cos = cos;
        if (ia == ib) r.argmax_agree += 1;
    }
    return r;
}

/// Metal bytes held by LIVE arrays, with the allocator's free list flushed so
/// the number is a function of ownership alone.
pub fn mlxSettledActiveBytes(s: mlx.mlx_stream) usize {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    return active;
}

/// `mlx_metal_is_available` is true for a Metal-enabled build with no usable
/// GPU, so this probes a real stream plus a 1x1 zeros+eval instead.
pub fn mlxDeviceUsable() bool {
    mlx.installErrorHandler();
    var trash: [256]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (s.ctx == null) {
        _ = mlx.takeError(&trash);
        return false;
    }
    const shape = [_]c_int{ 1, 1 };
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    if (mlx.mlx_zeros(&a, &shape, 2, .float32, s) != 0) {
        _ = mlx.takeError(&trash);
        return false;
    }
    if (mlx.mlx_array_eval(a) != 0) {
        _ = mlx.takeError(&trash);
        return false;
    }
    return true;
}

pub fn bf16BitsEqual(a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return false;
    const n = mlx.mlx_array_size(a);
    const pa = mlx.mlx_array_data_bfloat16(a) orelse return error.NoData;
    const pb = mlx.mlx_array_data_bfloat16(b) orelse return error.NoData;
    return std.mem.eql(u16, pa[0..n], pb[0..n]);
}

/// Whether two bf16 arrays hold the same bit patterns (`array_equal` takes -0 for +0).
pub fn bf16BitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var va = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(va);
    var vb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vb);
    try mlx.check(mlx.mlx_view(&va, a, .uint16, s));
    try mlx.check(mlx.mlx_view(&vb, b, .uint16, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, va, vb, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}
