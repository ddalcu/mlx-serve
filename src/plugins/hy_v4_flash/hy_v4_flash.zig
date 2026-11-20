//! hy_v4_flash (tencent/Hy4-flash) TEXT path, batch 1.
//!
//! Architecture source: huggingface/transformers PR #50903
//! (src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py). The text
//! trunk is DeepSeek-V4-flavoured (MLA, compressed indexer, sliding attention,
//! learnable sinks, hyper-connections, Engram, latent MoE) but differs from
//! deepseek_v41.zig in several contracts documented on the structs below.
//! All math here is f32, matching the reference and the NumPy oracle in
//! tests/hy4_flash_ref.py.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const engram_mod = @import("../deepseek_v41/engram.zig");
const config = @import("config.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const weights_load = @import("../../sdk/weights/load.zig");
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;

pub const Config = config.Config;
pub const parseConfigJson = config.parseConfigJson;
pub const parseRealConfig = config.parseRealConfig;

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const none: A = .{ .ctx = null };

/// Every intermediate of one forward is freed together. `out` detaches a
/// handle that must outlive the scope.
const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,

    fn init(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }
    fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }
    fn d(self: *Scope, name: []const u8, x: A) !void {
        _ = self;
        _ = name;
        _ = x;
    }

    fn di(self: *Scope, li: usize, suffix: []const u8, x: A) !void {
        _ = self;
        _ = li;
        _ = suffix;
        _ = x;
    }

    fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }
    fn res(self: *Scope, rc: c_int, o: *const A) !A {
        mlx.check(rc) catch |e| {
            _ = mlx.mlx_array_free(o.*);
            return e;
        };
        return self.keep(o.*);
    }
    fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }
    fn op1(sc: *Scope, comptime fun: anytype, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(fun(&o, x, sc.s), &o);
    }
    fn op2(sc: *Scope, comptime fun: anytype, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(fun(&o, x, y, sc.s), &o);
    }
    fn add(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_add, x, y);
    }
    fn sub(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_subtract, x, y);
    }
    fn mul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_multiply, x, y);
    }
    fn div(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_divide, x, y);
    }
    fn matmul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_matmul, x, y);
    }
    fn f(sc: *Scope, v: f32) !A {
        return sc.keep(mlx.mlx_array_new_float(v));
    }
    fn i(sc: *Scope, v: i32) !A {
        return sc.keep(mlx.mlx_array_new_int(v));
    }
    fn ints(sc: *Scope, data: []const i32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .int32));
    }
    fn floats(sc: *Scope, data: []const f32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
    }
    fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        if (mlx.mlx_array_dtype(x) == dt) return x;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn expand(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_expand_dims(&o, x, axis, sc.s), &o);
    }
    fn slice(sc: *Scope, x: A, axis_in: isize, lo: usize, hi: usize) !A {
        const sh = mlx.getShape(x);
        const axis: usize = if (axis_in < 0) @intCast(@as(isize, @intCast(sh.len)) + axis_in) else @intCast(axis_in);
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(stop[0..sh.len], sh);
        start[axis] = @intCast(lo);
        stop[axis] = @intCast(hi);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, sc.s), &o);
    }
    fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        if (xs.len == 1) return xs[0];
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    fn stack(sc: *Scope, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_stack_axis(&o, vec, axis, sc.s), &o);
    }
    fn take(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn takeAlong(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_along_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn reduce(sc: *Scope, comptime f_: anytype, x: A, axis: c_int, keep_: bool) !A {
        var o = mlx.mlx_array_new();
        return sc.res(f_(&o, x, axis, keep_, sc.s), &o);
    }
    fn sum(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_sum_axis, x, axis, keep_);
    }
    fn mean(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_mean_axis, x, axis, keep_);
    }
    fn max(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_max_axis, x, axis, keep_);
    }
    fn softmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_softmax_axis(&o, x, axis, true, sc.s), &o);
    }
    fn where(sc: *Scope, c: A, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_where(&o, c, x, y, sc.s), &o);
    }
    fn sqrt(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sqrt, x);
    }
    fn sigmoid(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sigmoid, x);
    }
    fn abs(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_abs, x);
    }
    fn sign(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sign, x);
    }
    fn maximum(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_maximum, x, y);
    }
    fn minimum(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_minimum, x, y);
    }
    fn arange(sc: *Scope, hi: usize) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_arange(&o, 0, @floatFromInt(hi), 1, .int32, sc.s), &o);
    }
    fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s), &o);
    }
    fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        const xf = try sc.astype(x, .float32);
        const sq = try sc.mul(xf, xf);
        const mn = try sc.mean(sq, -1, true);
        const ep = try sc.f(eps);
        const r = try sc.op1(mlx.mlx_rsqrt, try sc.add(mn, ep));
        return sc.mul(try sc.mul(xf, r), try sc.astype(w, .float32));
    }
    fn rmsUnweighted(sc: *Scope, x: A, eps: f32) !A {
        const xf = try sc.astype(x, .float32);
        const sq = try sc.mul(xf, xf);
        const mn = try sc.mean(sq, -1, true);
        const ep = try sc.f(eps);
        return sc.op1(mlx.mlx_rsqrt, try sc.add(mn, ep));
    }
};

fn dim(x: A, axis: isize) usize {
    const sh = mlx.getShape(x);
    const ax: usize = if (axis < 0) @intCast(@as(isize, @intCast(sh.len)) + axis) else @intCast(axis);
    return @intCast(sh[ax]);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

// ── weights ───────────────────────────────────────────────────────────────

const Layer = struct {
    attn_norm: A = none,
    ffn_norm: A = none,
    q_a: QLin = .{},
    q_norm: A = none,
    q_b: QLin = .{},
    kv_a: QLin = .{},
    kv_norm: A = none,
    kv_b: QLin = .{},
    gate: QLin = .{}, // attention output gate
    o: QLin = .{},
    post_norm: A = none,
    sink: A = none,
    hc_attn_fn: QLin = .{},
    hc_attn_base: A = none,
    hc_attn_scale: A = none,
    hc_ffn_fn: QLin = .{},
    hc_ffn_base: A = none,
    hc_ffn_scale: A = none,
    // MoE (null on dense layers)
    dense_gate: QLin = .{},
    dense_up: QLin = .{},
    dense_down: QLin = .{},
    dense_post: A = none,
    router: QLin = .{},
    router_bias: A = none,
    latent_down: QLin = .{},
    latent_up: QLin = .{},
    experts_gu: QLin = .{},
    experts_down: QLin = .{},
    shared_gate: QLin = .{},
    shared_up: QLin = .{},
    shared_down: QLin = .{},
    combine_norm: A = none,
    // Compressed indexer (null on sliding layers)
    iq: QLin = .{},
    ihw: QLin = .{},
    itk: QLin = .{},
    itk_norm: A = none,
    ikv: QLin = .{},
    igate: QLin = .{},
    ik_norm: A = none,
    iape: A = none,
};

const Engram = struct {
    li: usize,
    /// Dense fixture table; unused when the Model owns an external table.
    table: A = none,
    /// Global row of this layer's table in the shared external file.
    row_off: u64 = 0,
    lookup: A,
    heads: usize,
    head_dim: usize,
    mult: []const i64,
    sizes: []const usize,
    value_w: QLin = .{},
    value_bias: A = none,
    key_w: []QLin = &.{},
    key_bias: []A = &.{},
    key_norm: []A = &.{},
    query_norm: []A = &.{},
    per_head_norm: []A = &.{},
    pre_gamma: A = none,
    scales: []A = &.{},
};

fn dbgOn() bool {
    return std.c.getenv("HY4_DBG") != null;
}
var dbg_gu: usize = 0;
var dbg_dn: usize = 0;
fn dbgShape(label: []const u8, x: A) void {
    const sh = mlx.getShape(x);
    std.debug.print("[hy4dbg] {s} shape=[", .{label});
    for (sh, 0..) |d, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", d });
    std.debug.print("]\n", .{});
}

fn get(w: *const Weights, name: []const u8) !A {
    return w.get(name) orelse error.MissingWeight;
}

fn getf(a: std.mem.Allocator, w: *const Weights, comptime fmt: []const u8, args: anytype) !A {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    return get(w, name);
}

/// One affine linear. `w` is U32 packed [out, in*bits/32] (or [E, out, ...]
/// for an expert bank), `s`/`b` BF16 [.., in/gs]. `s == null` means a dense
/// weight, which is how the fp32 oracle fixture still runs through here.
const QuantParams = struct { bits: u32, group_size: u32, mode: QuantMode = .affine };

const QLin = struct {
    w: A = none,
    s: A = none,
    b: A = none,
    qp: QuantParams = .{ .bits = 0, .group_size = 0 },

    fn apply(self: *const QLin, sc: *Scope, x: A) !A {
        if (self.s.ctx == null) return sc.matmul(x, try sc.transpose(self.w, &.{ 1, 0 }));
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_quantized_matmul(&o, x, self.w, self.s, self.b, true,
            mlx.mlx_optional_int.some(@intCast(self.qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(self.qp.bits)),
            self.qp.mode.cstr(), sc.s), &o);
    }

    /// A handful of rows (embedding lookups), dequantized on the device.
    fn rows(self: *const QLin, sc: *Scope, ids: A, dt: mlx.mlx_dtype) !A {
        if (self.s.ctx == null) return sc.astype(try sc.take(self.w, ids, 0), dt);
        const qw = try sc.take(self.w, ids, 0);
        const qs = try sc.take(self.s, ids, 0);
        const qb: A = if (self.b.ctx != null) try sc.take(self.b, ids, 0) else none;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_dequantize(&o, qw, qs, qb,
            mlx.mlx_optional_int.some(@intCast(self.qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(self.qp.bits)),
            self.qp.mode.cstr(), none, .{ .value = dt, .has_value = true }, sc.s), &o);
    }
};

/// Loads `<base>.weight` and, when the weight is packed U32, its `.scales`
/// and `.biases` beside it. Bits are solved from the packed geometry against
/// the group size: `in = scales_cols * gs`, `bits = weight_cols * 32 / in`.
/// This is the same affine solve the other modules use, minus the import
/// cycle (transformer.zig imports this file).
fn getLin(a: std.mem.Allocator, w: *const Weights, gs: u32, comptime fmt: []const u8, args: anytype) !QLin {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const wt = try get(w, name);
    if (mlx.mlx_array_dtype(wt) != .uint32) return .{ .w = wt };
    if (!std.mem.endsWith(u8, name, ".weight")) return error.UnsupportedQuantizedLayout;
    const base = name[0 .. name.len - ".weight".len];
    const sn = try std.fmt.allocPrint(a, "{s}.scales", .{base});
    defer a.free(sn);
    const bn = try std.fmt.allocPrint(a, "{s}.biases", .{base});
    defer a.free(bn);
    const sc = w.get(sn) orelse return error.UnsupportedQuantizedLayout;
    const bi = w.get(bn) orelse return error.UnsupportedQuantizedLayout;
    const wsh = mlx.getShape(wt);
    const ssh = mlx.getShape(sc);
    if (wsh.len < 2 or ssh.len < 2 or gs == 0) return error.UnsupportedQuantizedLayout;
    const wcols: u32 = @intCast(wsh[wsh.len - 1]);
    const scols: u32 = @intCast(ssh[ssh.len - 1]);
    if (wcols == 0 or scols == 0) return error.UnsupportedQuantizedLayout;
    const in_dim = scols * gs;
    if (in_dim == 0 or (wcols * 32) % in_dim != 0) return error.UnsupportedQuantizedLayout;
    const bits = (wcols * 32) / in_dim;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return error.UnsupportedQuantizedLayout,
    }
    return .{ .w = wt, .s = sc, .b = bi, .qp = .{ .bits = bits, .group_size = gs } };
}

/// A gathered affine expert bank: `x` [T,k,1,in], `idx` [T,k] -> [T,k,1,out],
/// the same contract dsv41's `routed` uses for `mlx_gather_qmm`.
fn gatherMm(sc: *Scope, x: A, b: *const QLin, idx: A) !A {
    var o = mlx.mlx_array_new();
    if (b.s.ctx == null) {
        const tr = try sc.transpose(b.w, &.{ 0, 2, 1 });
        return sc.res(mlx.mlx_gather_mm(&o, x, tr, none, idx, false, sc.s), &o);
    }
    return sc.res(mlx.mlx_gather_qmm(&o, x, b.w, b.s, b.b, none, idx, true,
        mlx.mlx_optional_int.some(@intCast(b.qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(b.qp.bits)),
        b.qp.mode.cstr(), false, sc.s), &o);
}

// ── rope ──────────────────────────────────────────────────────────────────

const Rope = struct { cos: A, sin: A };

fn buildRope(a: std.mem.Allocator, base: f64, rope_dim: usize, mrope: [3]usize, upto: usize) !Rope {
    const half = rope_dim / 2;
    const cs = try a.alloc(f32, upto * rope_dim);
    const sn = try a.alloc(f32, upto * rope_dim);
    for (0..upto) |pos| {
        for (0..half) |j| {
            const freq = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(rope_dim)));
            const c: f32 = @floatCast(@cos(freq));
            const si: f32 = @floatCast(@sin(freq));
            const plan = [3]usize{ 12, mrope[1], mrope[2] };
            _ = plan;
            // The reference's `repeat_interleave(2)` gives an interleaved
            // even/odd layout: both entries of a pair share the same phase.
            cs[pos * rope_dim + 2 * j] = c;
            cs[pos * rope_dim + 2 * j + 1] = c;
            sn[pos * rope_dim + 2 * j] = si;
            sn[pos * rope_dim + 2 * j + 1] = si;
        }
        // Overwrite the height/width lanes of the first half, as the
        // reference's recomposition_frequencies does.
        for (1..3) |axis| {
            const len = mrope[axis] * 3;
            var idx = axis;
            while (idx < len and idx < half) : (idx += 3) {
                const f1 = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * idx)) / @as(f64, @floatFromInt(rope_dim)));
                cs[pos * rope_dim + 2 * idx] = @floatCast(@cos(f1));
                cs[pos * rope_dim + 2 * idx + 1] = @floatCast(@cos(f1));
                sn[pos * rope_dim + 2 * idx] = @floatCast(@sin(f1));
                sn[pos * rope_dim + 2 * idx + 1] = @floatCast(@sin(f1));
            }
        }
    }
    const shape = [_]c_int{ ci(upto), ci(rope_dim) };
    defer a.free(cs);
    defer a.free(sn);
    return .{
        .cos = try keepArray(mlx.mlx_array_new_data(cs.ptr, &shape, 2, .float32)),
        .sin = try keepArray(mlx.mlx_array_new_data(sn.ptr, &shape, 2, .float32)),
    };
}

fn keepArray(x: A) !A {
    if (x.ctx == null) return error.MlxError;
    return x;
}

// ── state (per request, owned by the module) ─────────────────────────────

const LayerCache = struct {
    kp: ?A = null,
    kr: ?A = null,
    tk: ?A = null,
    fn clear(self: *LayerCache) void {
        if (self.kp) |x| _ = mlx.mlx_array_free(x);
        if (self.kr) |x| _ = mlx.mlx_array_free(x);
        if (self.tk) |x| _ = mlx.mlx_array_free(x);
        self.* = .{};
    }
};

pub const State = struct {
    a: std.mem.Allocator,
    s: S,
    caches: []LayerCache,
    len: usize = 0,
    /// Tokens immediately before the current forward's chunk, newest first:
    /// hist[0] is t-1, hist[1] is t-2. Engram n-grams need these on decode.
    hist: []u32 = &.{},

    pub fn init(a: std.mem.Allocator, n: usize, s: S, pad: u32, history: usize) !State {
        const cs = try a.alloc(LayerCache, n);
        @memset(cs, .{});
        const h = try a.alloc(u32, @max(1, history));
        @memset(h, pad);
        return .{ .a = a, .s = s, .caches = cs, .hist = h };
    }
    pub fn deinit(self: *State) void {
        for (self.caches) |*c| c.clear();
        self.a.free(self.caches);
        self.a.free(self.hist);
    }
    pub fn reset(self: *State) void {
        for (self.caches) |*c| c.clear();
        self.len = 0;
    }
};

// ── model ─────────────────────────────────────────────────────────────────

pub const Model = struct {
    a: std.mem.Allocator,
    s: S,
    cfg: Config,
    w: *const Weights,
    embed: QLin,
    head: QLin,
    norm: A,
    hc_fn: QLin,
    hc_base: A,
    hc_scale: A,
    /// The shared external Engram table, when the pack externalized it.
    ext_table: ?engram_mod.Table = null,
    layers: []Layer,
    engrams: []Engram,
    rope_indexed: Rope,
    rope_swa: Rope,
    state: State,
    /// Effective MoE top-k, overridable for the bandwidth diagnostic.
    topk_eff: usize,

    pub fn init(a: std.mem.Allocator, cfg: Config, w: *const Weights, s: S) !Model {
        var topk_eff = cfg.top_k;
        if (std.c.getenv("HY4_TOPK")) |v| {
            if (std.fmt.parseInt(usize, std.mem.span(v), 10)) |n| {
                if (n > 0 and n <= 512) topk_eff = n;
            } else |_| {}
        }
        var layers = try a.alloc(Layer, cfg.layers);
        for (layers) |*l| l.* = .{};
        for (0..cfg.layers) |li| {
            const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{li});
            defer a.free(p);
            const ax = try std.fmt.allocPrint(a, "{s}self_attn.", .{p});
            defer a.free(ax);
            const hc = try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.", .{p});
            defer a.free(hc);
            const hm = try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.", .{p});
            defer a.free(hm);
            const l = &layers[li];
            l.attn_norm = try getf(a, w, "{s}input_layernorm.weight", .{p});
            l.ffn_norm = try getf(a, w, "{s}post_attention_layernorm.weight", .{p});
            l.q_a = try getLin(a, w, cfg.quant_group_size, "{s}q_a_proj.weight", .{ax});
            l.q_norm = try getf(a, w, "{s}q_a_layernorm.weight", .{ax});
            l.q_b = try getLin(a, w, cfg.quant_group_size, "{s}q_b_proj.weight", .{ax});
            l.kv_a = try getLin(a, w, cfg.quant_group_size, "{s}kv_a_proj_with_mqa.weight", .{ax});
            l.kv_norm = try getf(a, w, "{s}kv_a_layernorm.weight", .{ax});
            l.kv_b = try getLin(a, w, cfg.quant_group_size, "{s}kv_b_proj.weight", .{ax});
            l.gate = try getLin(a, w, cfg.quant_group_size, "{s}linear_gate.weight", .{ax});
            l.o = try getLin(a, w, cfg.quant_group_size, "{s}o_proj.weight", .{ax});
            l.post_norm = try getf(a, w, "{s}post_proj_norm.weight", .{ax});
            l.sink = try getf(a, w, "{s}learnable_sink_param", .{ax});
            l.hc_attn_fn = try getLin(a, w, cfg.quant_group_size, "{s}hc_fn", .{hc});
            l.hc_attn_base = try getf(a, w, "{s}hc_base", .{hc});
            l.hc_attn_scale = try getf(a, w, "{s}hc_scale", .{hc});
            l.hc_ffn_fn = try getLin(a, w, cfg.quant_group_size, "{s}hc_fn", .{hm});
            l.hc_ffn_base = try getf(a, w, "{s}hc_base", .{hm});
            l.hc_ffn_scale = try getf(a, w, "{s}hc_scale", .{hm});
            const is_sparse = if (li < cfg.sparse.len) cfg.sparse[li] else true;
            if (is_sparse) {
                const mp = try std.fmt.allocPrint(a, "{s}mlp.", .{p});
                defer a.free(mp);
                l.router = try getLin(a, w, cfg.quant_group_size, "{s}gate.weight", .{mp});
                l.router_bias = try getf(a, w, "{s}gate.e_score_correction_bias", .{mp});
                l.latent_down = try getLin(a, w, cfg.quant_group_size, "{s}routed_expert_down_proj.weight", .{mp});
                l.latent_up = try getLin(a, w, cfg.quant_group_size, "{s}routed_expert_up_proj.weight", .{mp});
                l.experts_gu = try getLin(a, w, cfg.quant_group_size, "{s}experts.gate_up_proj.weight", .{mp});
                l.experts_down = try getLin(a, w, cfg.quant_group_size, "{s}experts.down_proj.weight", .{mp});
                l.shared_gate = try getLin(a, w, cfg.quant_group_size, "{s}shared_experts.gate_proj.weight", .{mp});
                l.shared_up = try getLin(a, w, cfg.quant_group_size, "{s}shared_experts.up_proj.weight", .{mp});
                l.shared_down = try getLin(a, w, cfg.quant_group_size, "{s}shared_experts.down_proj.weight", .{mp});
                l.combine_norm = try getf(a, w, "{s}combine_output_norm.weight", .{mp});
            } else {
                const mp = try std.fmt.allocPrint(a, "{s}mlp.", .{p});
                defer a.free(mp);
                l.dense_gate = try getLin(a, w, cfg.quant_group_size, "{s}gate_proj.weight", .{mp});
                l.dense_up = try getLin(a, w, cfg.quant_group_size, "{s}up_proj.weight", .{mp});
                l.dense_down = try getLin(a, w, cfg.quant_group_size, "{s}down_proj.weight", .{mp});
                l.dense_post = try getf(a, w, "{s}post_fc2_norm.weight", .{mp});
            }
            const is_indexed = if (li < cfg.indexed.len) cfg.indexed[li] else true;
            if (is_indexed) {
                const ip = try std.fmt.allocPrint(a, "{s}compressed_indexer.", .{ax});
                defer a.free(ip);
                l.iq = try getLin(a, w, cfg.quant_group_size, "{s}query_projection.weight", .{ip});
                l.ihw = try getLin(a, w, cfg.quant_group_size, "{s}head_weight_projection.weight", .{ip});
                l.itk = try getLin(a, w, cfg.quant_group_size, "{s}token_key_projection.weight", .{ip});
                l.itk_norm = try getf(a, w, "{s}token_key_norm.weight", .{ip});
                l.ikv = try getLin(a, w, cfg.quant_group_size, "{s}key_value_projection.weight", .{ip});
                l.igate = try getLin(a, w, cfg.quant_group_size, "{s}key_gate_projection.weight", .{ip});
                l.ik_norm = try getf(a, w, "{s}key_norm.weight", .{ip});
                l.iape = try getf(a, w, "{s}intra_group_ape", .{ip});
            }
        }

        var engrams = try a.alloc(Engram, cfg.engram_layers.len);
        for (engrams) |*e| e.* = undefined;
        for (cfg.engram_layers, 0..) |li, k| {
            const p = try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.", .{li});
            defer a.free(p);
            const per_layer = sizesPerLayer(cfg.engram_sizes, k);
            var nheads: usize = 0;
            for (per_layer) |_| nheads += 1;
            var e = &engrams[k];
            e.li = li;
            e.sizes = per_layer;
            // The tiny oracle carries its multipliers in the config; the real
            // checkpoint's come from numpy default_rng(seed + 10007*layer_id)
            // (seed=0, bound=i64max/lookup_size/2), precomputed here.
            e.mult = if (cfg.hash_mult.len >= cfg.max_ngram)
                cfg.hash_mult
            else
                hashMultipliers(li + 1);
            // A real pack externalizes this table; the dense fixture keeps it
            // in the shard. Row offsets follow the table order in the config's
            // `engram_table` spec (layers are 0-based).
            if (cfg.engram_table) |spec| {
                var found = false;
                for (spec.layers, 0..) |l, idx| if (l == li) {
                    e.row_off = spec.rowOffset(idx);
                    found = true;
                    break;
                };
                if (!found and spec.layers.len == 0 and k < spec.table_rows.len) e.row_off = spec.rowOffset(k);
            } else {
                e.table = try getf(a, w, "{s}multi_head_embedding.embedding.weight", .{p});
            }
            e.lookup = try getf(a, w, "{s}lookup_table", .{p});
            e.heads = nheads;
            e.head_dim = cfg.head_dim;
            e.value_w = try getLin(a, w, cfg.quant_group_size, "{s}value_proj.weight", .{p});
            const value_bias_name = try std.fmt.allocPrint(a, "{s}value_proj.bias", .{p});
            defer a.free(value_bias_name);
            e.value_bias = w.get(value_bias_name) orelse none;
            e.key_w = try a.alloc(QLin, cfg.hc_mult);
            e.key_bias = try a.alloc(A, cfg.hc_mult);
            e.key_norm = try a.alloc(A, cfg.hc_mult);
            e.query_norm = try a.alloc(A, cfg.hc_mult);
            e.per_head_norm = try a.alloc(A, nheads);
            e.scales = try a.alloc(A, cfg.hc_mult);
            for (0..cfg.hc_mult) |h| {
                e.key_w[h] = try getLin(a, w, cfg.quant_group_size, "{s}key_projs.{d}.weight", .{ p, h });
                const key_bias_name = try std.fmt.allocPrint(a, "{s}key_projs.{d}.bias", .{ p, h });
                defer a.free(key_bias_name);
                e.key_bias[h] = w.get(key_bias_name) orelse none;
                e.key_norm[h] = try getf(a, w, "{s}norm_key.{d}.weight", .{ p, h });
                e.query_norm[h] = try getf(a, w, "{s}norm_query.{d}.weight", .{ p, h });
                e.scales[h] = try getf(a, w, "{s}unified_per_hc_scales.{d}", .{ p, h });
            }
            for (0..nheads) |h| e.per_head_norm[h] = try getf(a, w, "{s}lookup_per_head_norms.{d}.weight", .{ p, h });
            e.pre_gamma = try getf(a, w, "{s}unified_pre_gamma", .{p});
        }

        var ext_table: ?engram_mod.Table = null;
        errdefer if (ext_table) |*t| t.close();
        if (cfg.engram_table) |spec| ext_table = try engram_mod.Table.openHy4(a, cfg.model_dir, &spec);
        return .{
            .a = a,
            .s = s,
            .cfg = cfg,
            .w = w,
            .embed = try getLin(a, w, cfg.quant_group_size, "model.embed_tokens.weight", .{}),
            .head = try getLin(a, w, cfg.quant_group_size, "lm_head.weight", .{}),
            .norm = try get(w, "model.norm.weight"),
            .hc_fn = try getLin(a, w, cfg.quant_group_size, "model.hc_head.hc_head_fn", .{}),
            .hc_base = try get(w, "model.hc_head.hc_head_base"),
            .hc_scale = try get(w, "model.hc_head.hc_head_scale"),
            .ext_table = ext_table,
            .layers = layers,
            .engrams = engrams,
            .rope_indexed = try buildRope(a, cfg.indexed_theta, cfg.rope_dim, cfg.mrope, 32768),
            .rope_swa = try buildRope(a, cfg.swa_theta, cfg.rope_dim, cfg.mrope, 32768),
            .state = try State.init(a, cfg.layers, s, @intCast(cfg.pad), @max(2, cfg.max_ngram -| 1)),
            .topk_eff = topk_eff,
        };
    }

    pub fn deinit(self: *Model) void {
        self.state.deinit();
        if (self.ext_table) |*t| t.close();
        for (self.engrams) |e| {
            self.a.free(e.key_w);
            self.a.free(e.key_bias);
            self.a.free(e.key_norm);
            self.a.free(e.query_norm);
            self.a.free(e.per_head_norm);
            self.a.free(e.scales);
        }
        self.a.free(self.engrams);
        self.a.free(self.layers);
        _ = mlx.mlx_array_free(self.rope_indexed.cos);
        _ = mlx.mlx_array_free(self.rope_indexed.sin);
        _ = mlx.mlx_array_free(self.rope_swa.cos);
        _ = mlx.mlx_array_free(self.rope_swa.sin);
    }

    fn lin(sc: *Scope, x: A, l: QLin) !A {
        return l.apply(sc, x);
    }

    fn ropeTake(sc: *Scope, rope: Rope, pos_lo: usize, pos_hi: usize) !struct { cos: A, sin: A } {
        return .{
            .cos = try sc.slice(rope.cos, 0, pos_lo, pos_hi),
            .sin = try sc.slice(rope.sin, 0, pos_lo, pos_hi),
        };
    }

    fn applyRope(sc: *Scope, q: A, cos: A, sin: A) !A {
        return ropeRotate(sc, q, cos, sin);
    }

    fn hci(sc: *Scope, streams: A, fn_w: QLin, base: A, scale: A, cfg: Config) !struct { post: A, out: A } {
        const T = dim(streams, 0);
        const flat = try sc.reshape(streams, &.{ ci(T), ci(cfg.hc_mult * cfg.hidden) });
        const inv = try sc.rmsUnweighted(flat, cfg.eps);
        const mixes = try sc.mul(try lin(sc, flat, fn_w), inv);
        const pre = try sc.slice(mixes, -1, 0, cfg.hc_mult);
        const post_l = try sc.slice(mixes, -1, cfg.hc_mult, 2 * cfg.hc_mult);
        const sc0 = try sc.slice(scale, -1, 0, 1);
        const sc1 = try sc.slice(scale, -1, 1, 2);
        const b0 = try sc.slice(base, -1, 0, cfg.hc_mult);
        const b1 = try sc.slice(base, -1, cfg.hc_mult, 2 * cfg.hc_mult);
        const eps = try sc.f(cfg.hc_eps);
        const pre_g = try sc.add(try sc.sigmoid(try sc.add(try sc.mul(pre, sc0), b0)), eps);
        const post_g = try sc.add(try sc.mul(try sc.f(cfg.hc_magnitude), try sc.sigmoid(try sc.add(try sc.mul(post_l, sc1), b1))), eps);
        const out = try sc.sum(try sc.mul(try sc.expand(pre_g, 2), streams), 1, false);
        return .{ .post = post_g, .out = out };
    }

    fn attention(self: *Model, sc: *Scope, li: usize, x: A, q_resid_out: *?A, st: *State, pos_lo: usize) !A {
        const cfg = self.cfg;
        const l = &self.layers[li];
        const T = dim(x, 0);
        const indexed = cfg.indexed[li];
        const heads = if (indexed) cfg.indexed_heads else cfg.swa_heads;
        const nope = if (indexed) cfg.qk_nope else cfg.swa_qk_nope;
        const vd = if (indexed) cfg.v_dim else cfg.swa_v_dim;
        const qd = nope + cfg.rope_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qd)));

        const gate = try sc.reshape(try lin(sc, x, l.gate), &.{ ci(T), ci(heads), ci(vd) });
        if (li <= 1 or li == 21) try sc.di(li, "linear_gate", gate);
        const qa = try lin(sc, x, l.q_a);
        if (li <= 1 or li == 21) try sc.di(li, "q_a_proj", qa);
        const q_resid = try sc.rms(qa, l.q_norm, cfg.eps);
        if (li <= 1 or li == 21) try sc.di(li, "q_a_layernorm", q_resid);
        q_resid_out.* = q_resid;
        const q = try sc.reshape(try lin(sc, q_resid, l.q_b), &.{ ci(T), ci(heads), ci(qd) });
        if (li <= 1 or li == 21) try sc.di(li, "q_b_proj", q);
        const qp = try sc.slice(q, -1, 0, nope);
        const qr = try sc.slice(q, -1, nope, qd);

        const compressed = try lin(sc, x, l.kv_a);
        if (li <= 1 or li == 21) try sc.di(li, "kv_a_proj_with_mqa", compressed);
        const kvp = try sc.slice(compressed, -1, 0, cfg.kv_lora);
        const kr0 = try sc.slice(compressed, -1, cfg.kv_lora, cfg.kv_lora + cfg.rope_dim);
        const kp_new = try sc.rms(kvp, l.kv_norm, cfg.eps);
        if (li <= 1 or li == 21) try sc.di(li, "kv_a_layernorm", kp_new);
        const rope = if (indexed) self.rope_indexed else self.rope_swa;
        const rp = try ropeTake(sc, rope, pos_lo, pos_lo + T);
        const kr_new = try applyRope(sc, kr0, rp.cos, rp.sin);
        if (li <= 1 or li == 21) try sc.di(li, "rope_cos", rp.cos);
        if (li <= 1 or li == 21) try sc.di(li, "rope_sin", rp.sin);
        if (li <= 1 or li == 21) try sc.di(li, "kr_rope", kr_new);

        const cache = &st.caches[li];
        const kp_all = if (cache.kp) |old| try sc.concat(&.{ old, kp_new }, 0) else kp_new;
        const kr_all = if (cache.kr) |old| try sc.concat(&.{ old, kr_new }, 0) else kr_new;
        // The cached values must outlive this scope.
        const kp_keep = sc.out(kp_all);
        const kr_keep = sc.out(kr_all);
        cache.kp = kp_keep;
        cache.kr = kr_keep;
        const S_all = dim(kp_all, 0);

        const kv = try sc.reshape(try lin(sc, kp_all, l.kv_b), &.{ ci(S_all), ci(heads), ci(nope + vd) });
        if (li <= 1 or li == 21) try sc.di(li, "kv_b_proj", kv);
        const kn = try sc.slice(kv, -1, 0, nope);
        const vv = try sc.slice(kv, -1, nope, nope + vd);
        const kr_exp = try sc.broadcast(try sc.expand(kr_all, 1), &.{ ci(S_all), ci(heads), ci(cfg.rope_dim) });
        const k = try sc.concat(&.{ kn, kr_exp }, -1);
        const qr_e = try sc.concat(&.{ qp, try applyRope(sc, qr, rp.cos, rp.sin) }, -1);
        if (li <= 1 or li == 21) try sc.di(li, "q_rope", qr_e);

        // scores [heads, T, S]
        const qt = try sc.transpose(qr_e, &.{ 1, 0, 2 });
        const kt = try sc.transpose(k, &.{ 1, 2, 0 });
        var scores = try sc.matmul(qt, kt);
        if (li <= 1 or li == 21) try sc.di(li, "scores_raw", scores);
        const sf = try sc.f(scale);
        scores = try sc.mul(scores, sf);
        if (li <= 1 or li == 21) try sc.di(li, "scores_scaled", scores);

        // Causal + sliding mask, then optional indexer selection.
        const qpos = try sc.expand(try sc.slice(try sc.arange(S_all), 0, S_all - T, S_all), 1);
        const kpos = try sc.expand(try sc.arange(S_all), 0);
        const causal = try sc.op2(mlx.mlx_less_equal, kpos, qpos);
        const neg = try sc.f(-1e30);
        scores = try sc.where(causal, scores, neg);
        if (!indexed) {
            const win = try sc.sub(qpos, try sc.f(@as(f32, @floatFromInt(cfg.sliding)) - 1));
            const keep = try sc.op2(mlx.mlx_greater_equal, kpos, win);
            scores = try sc.where(keep, scores, neg);
        } else if (S_all > cfg.candidate_topk and std.c.getenv("HY4_NO_INDEXER") == null) {
            // Ablation arm 1: with HY4_NO_INDEXER=1 the indexed layers use
            // plain dense causal attention instead of the indexer keep mask.
            scores = try self.indexMask(sc, li, x, q_resid, S_all, T, qpos, scores, st);
        }

        // Sinks join the softmax denominator (per head). scores [H,T,S].
        const sinks = try sc.reshape(try sc.astype(l.sink, .float32), &.{ ci(heads), ci(1), ci(1) });
        const mx = try sc.maximum(try sc.max(scores, -1, true), sinks);
        const ex = try sc.op1(mlx.mlx_exp, try sc.sub(scores, mx));
        const es = try sc.op1(mlx.mlx_exp, try sc.sub(sinks, mx));
        const den = try sc.add(try sc.sum(ex, -1, true), es);
        const probs = try sc.div(ex, den);
        const vt = try sc.transpose(vv, &.{ 1, 0, 2 });
        const o = try sc.transpose(try sc.matmul(probs, vt), &.{ 1, 0, 2 });
        if (li <= 1 or li == 21) try sc.di(li, "attn_output", o);
        const gated = try sc.mul(o, try sc.sigmoid(gate));
        const flat = try sc.reshape(gated, &.{ ci(T), ci(heads * vd) });
        const proj = try lin(sc, flat, l.o);
        if (li <= 1 or li == 21) try sc.di(li, "o_proj", proj);
        return sc.rms(proj, l.post_norm, cfg.eps);
    }

    fn indexMask(self: *Model, sc: *Scope, li: usize, x: A, q_resid: A, S_all: usize, T: usize, qpos: A, scores: A, st: *State) !A {
        const cfg = self.cfg;
        const l = &self.layers[li];
        const ratio = cfg.ratio;
        const G = S_all / ratio;
        const tk_new = try sc.rms(try lin(sc, x, l.itk), l.itk_norm, cfg.eps);
        const cache = &st.caches[li];
        const tk_all = if (cache.tk) |old| try sc.concat(&.{ old, tk_new }, 0) else tk_new;
        cache.tk = sc.out(tk_all);
        const grouped = try sc.reshape(try sc.slice(tk_all, 0, 0, G * ratio), &.{ ci(G), ci(ratio), ci(cfg.idx_head_dim) });
        const gf = try sc.astype(grouped, .float32);
        const val = try lin(sc, gf, l.ikv);
        const gat = try sc.add(try lin(sc, gf, l.igate), try sc.astype(l.iape, .float32));
        const gm = try sc.sub(gat, try sc.max(gat, 1, true));
        const ge = try sc.op1(mlx.mlx_exp, gm);
        const gw = try sc.div(ge, try sc.sum(ge, 1, true));
        const pooled = try sc.rms(try sc.sum(try sc.mul(val, gw), 1, true), l.ik_norm, cfg.eps);
        const q = try sc.reshape(try lin(sc, q_resid, l.iq), &.{ ci(T), ci(cfg.idx_heads), ci(cfg.idx_head_dim) });
        const rope = self.rope_indexed;
        const qp = try ropeTake(sc, rope, S_all - T, S_all);
        const kp = try ropeTake(sc, rope, 0, G * ratio);
        // Query and key use the group-start positions for keys.
        const q_rot = try partialRope(sc, q, qp.cos, qp.sin, cfg.rope_dim);
        // Pooling keeps a size-1 axis, the reference's group-key head axis.
        const k_rot = try partialRope(sc, pooled, try sliceEvery(sc, kp.cos, ratio, G), try sliceEvery(sc, kp.sin, ratio, G), cfg.rope_dim);
        const hw = try sc.mul(try lin(sc, x, l.ihw), try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(cfg.idx_heads * cfg.idx_head_dim)))));
        // relu scores [H, T, G], head weights [H, T] -> logits [T, G].
        const qt = try sc.transpose(q_rot, &.{ 1, 0, 2 });
        const kt = try sc.transpose(k_rot, &.{ 1, 2, 0 });
        const raw = try sc.matmul(qt, kt);
        const relu = try sc.maximum(raw, try sc.f(0));
        const hw_t = try sc.expand(try sc.transpose(hw, &.{ 1, 0 }), -1);
        const logits = try sc.sum(try sc.mul(relu, hw_t), 0, false);
        const gidx = try sc.arange(G);
        const own = try sc.op2(mlx.mlx_floor_divide, qpos, try sc.f(@as(f32, @floatFromInt(ratio))));
        const allowed = try sc.op2(mlx.mlx_less, gidx, own);
        const ng = try sc.f(-1e30);
        const logits2 = try sc.where(allowed, logits, ng);
        // top (group_topk - 1) groups, plus the group the query sits in.
        const kkeep = @min(cfg.group_topk - 1, G);
        const topidx = try topkIdx(sc, logits2, kkeep);
        const own_i = try sc.astype(own, .int32);
        const all = try sc.concat(&.{ topidx, own_i }, -1);
        // Expand groups to token slots, then one-hot them into [T, S] with
        // an equality over every selected slot (no scatter in mlx-c).
        const offs = try sc.arange(ratio);
        const slots_f = try sc.add(try sc.mul(try sc.expand(all, -1), try sc.f(@as(f32, @floatFromInt(ratio)))), offs);
        const slots = try sc.reshape(slots_f, &.{ ci(T), ci(dim(all, -1) * ratio) });
        const slotsI = try sc.astype(slots, .int32);
        const ar = try sc.arange(S_all);
        const eq = try sc.op2(mlx.mlx_equal, try sc.expand(slotsI, -1), try sc.expand(ar, 0));
        const eqf = try sc.astype(eq, .float32);
        const marked = try sc.max(eqf, 1, false);
        return sc.where(try sc.expand(marked, 0), scores, ng);
    }

    fn mlp(self: *Model, sc: *Scope, li: usize, x: A) !A {
        const cfg = self.cfg;
        const l = &self.layers[li];
        const is_sparse = if (li < cfg.sparse.len) cfg.sparse[li] else true;
        if (!is_sparse) {
            const h = try lin(sc, x, l.dense_gate);
            const u = try lin(sc, x, l.dense_up);
            if (li <= 1 or li == 21) {
                try sc.di(li, "dense_gate", h);
                try sc.di(li, "dense_up", u);
            }
            const act = try sc.mul(try sc.mul(h, try sc.sigmoid(h)), u);
            if (li <= 1 or li == 21) try sc.di(li, "dense_act", act);
            const dn = try lin(sc, act, l.dense_down);
            if (li <= 1 or li == 21) try sc.di(li, "dense_down", dn);
            return sc.rms(dn, l.dense_post, cfg.eps);
        }
        const T = dim(x, 0);
        const logits = try sc.astype(try lin(sc, x, l.router), .float32);
        const sg = try sc.sigmoid(logits);
        const choice = try sc.add(sg, try sc.astype(l.router_bias, .float32));
        const top = try topkIdx(sc, choice, self.topk_eff);
        if (li <= 1 or li == 21) { try sc.di(li, "router_logits", logits); try sc.di(li, "router_sg", sg); try sc.di(li, "router_choice", choice); try sc.di(li, "moe_topk", top); }
        const tw = try sc.takeAlong(sg, top, -1);
        const den = try sc.add(try sc.sum(tw, -1, true), try sc.f(1e-20));
        const nw = try sc.mul(try sc.div(tw, den), try sc.f(cfg.route_scale));
        if (li <= 1 or li == 21) { try sc.di(li, "moe_tw", tw); try sc.di(li, "moe_nw", nw); }
        const latent = try lin(sc, x, l.latent_down);
        if (li <= 1 or li == 21) try sc.di(li, "moe_latent", latent);
        const gu = try gatherGu(sc, l.experts_gu, latent, top, cfg);
        if (li <= 1 or li == 21) try sc.di(li, "moe_gu", gu);
        const lim = cfg.swiglu_limit;
        const half = cfg.moe_inter;
        const g = try sc.slice(gu, -1, 0, half);
        const u = try sc.slice(gu, -1, half, half * 2);
        const gc = try sc.minimum(g, try sc.f(lim));
        const uc = try sc.minimum(try sc.maximum(u, try sc.f(-lim)), try sc.f(lim));
        const act = try sc.mul(try sc.mul(gc, try sc.sigmoid(gc)), uc);
        if (li <= 1 or li == 21) try sc.di(li, "moe_act", act);
        const dn = try gatherDown(sc, l.experts_down, act, top, cfg);
        if (li <= 1 or li == 21) try sc.di(li, "moe_dn", dn);
        const routed = try sc.mul(dn, try sc.expand(nw, -1));
        const routed_h = try lin(sc, try sc.sum(routed, 1, false), l.latent_up);
        if (li <= 1 or li == 21) { try sc.di(li, "moe_routed", routed); try sc.di(li, "moe_routed_h", routed_h); }
        const sh = try lin(sc, x, l.shared_gate);
        const su = try lin(sc, x, l.shared_up);
        const shared = try lin(sc, try sc.mul(try sc.mul(sh, try sc.sigmoid(sh)), su), l.shared_down);
        if (li <= 1 or li == 21) { try sc.di(li, "moe_shared_gate", sh); try sc.di(li, "moe_shared_up", su); try sc.di(li, "moe_shared", shared); }
        _ = T;
        const comb = try sc.rms(try sc.add(routed_h, shared), l.combine_norm, cfg.eps);
        if (li <= 1 or li == 21) try sc.di(li, "moe_combine", comb);
        return comb;
    }

    fn engram(self: *Model, sc: *Scope, li: usize, streams: A, ids: []const u32, st: *State) !A {
        const cfg = self.cfg;
        var en: ?*Engram = null;
        for (self.engrams) |*x| if (x.li == li) {
            en = x;
            break;
        };
        const e = en orelse return streams;
        const T = dim(streams, 0);
        const nh = e.heads;
        const sizes = e.sizes;
        // The normalized-vocabulary map is tiny on the read side (one int per
        // token id); read it once to host and hash there (mlx-c has no xor).
        try mlx.check(mlx.mlx_array_eval(e.lookup));
        // A real pack ships the lookup table as I32; the fp32 fixture wrote it
        // as F32. mlx_array_data_float32 happily returns the raw pointer of an
        // I32 array, so the dtype must decide the read, not pointer presence.
        const lut_i32: ?[*]const i32 = if (mlx.mlx_array_dtype(e.lookup) == .int32) mlx.mlx_array_data_int32(e.lookup) else null;
        const lut_f32: ?[*]const f32 = if (mlx.mlx_array_dtype(e.lookup) == .float32) mlx.mlx_array_data_float32(e.lookup) else null;
        if (lut_i32 == null and lut_f32 == null) return error.NoData;
        const hids = try self.a.alloc(i32, T * nh);
        defer self.a.free(hids);
        for (0..T) |t| {
            var head: usize = 0;
            var off: i64 = 0;
            const per_order = nh / @max(1, cfg.max_ngram - 1);
            for (0..cfg.max_ngram - 1) |oi| {
                const mods = sizes[oi * per_order .. (oi + 1) * per_order];
                var mix: i64 = 0;
                for (0..oi + 2) |slot| {
                    // Reference build_ngram_tokens uses the full sequence and
                    // position_ids: for t-slot before this chunk, read the
                    // carried history. hist[s-t-1] is exactly that token.
                    const p: u32 = if (slot <= t) ids[t - slot] else st.hist[slot - t - 1];
                    const n = mlx.mlx_array_size(e.lookup);
                    // Reference clamps to table_size-1 and substitutes the
                    // pad row for ids past the compressed vocabulary.
                    const ix: usize = if (p < n) @intCast(p) else cfg.pad;
                    const compressed: i64 = if (lut_i32) |lp|
                        @as(i64, lp[ix])
                    else
                        @as(i64, @intFromFloat(lut_f32.?[ix]));
                    const m = e.mult[slot];
                    const term = compressed * m;
                    if (slot == 0) mix = term else mix ^= term;
                }
                for (mods) |modulus| {
                    hids[t * nh + head] = @intCast(@mod(mix, @as(i64, @intCast(modulus))) + off);
                    head += 1;
                    off += @intCast(modulus);
                }
            }
        }
        var emb: A = none;
        if (self.ext_table) |*table| {
            // hids are row ids local to this layer's table; the shared file
            // appends the four tables in config order, so add this layer's
            // base row. The host gather + pread pool are dsv41_engram's.
            const ids_u32 = try self.a.alloc(u32, T * nh);
            defer self.a.free(ids_u32);
            for (hids, 0..) |h, i| ids_u32[i] = @intCast(@as(i64, h) + @as(i64, @intCast(e.row_off)));
            const raw = try table.read(self.a, ids_u32);
            defer raw.deinit(self.a);
            const flat = try sc.keep(try table.dequant(raw, ids_u32.len, sc.s));
            emb = try sc.reshape(flat, &.{ ci(T), ci(nh), ci(e.head_dim) });
        } else {
            const hidA = try sc.ints(hids, &.{ ci(T), ci(nh) });
            emb = try sc.take(e.table, hidA, 0);
        }
        if (li == 1 or li == 21) try sc.di(li, "engram_emb", emb);
        if (cfg.engram_fp8) {
            // Per-head E4M3 round trip used when the checkpoint enables it.
            const scale = try sc.div(try sc.maximum(try sc.max(try sc.abs(emb), -1, true), try sc.f(1e-12)), try sc.f(448));
            var codes = mlx.mlx_array_new();
            const normalized = try sc.div(emb, scale);
            const codes_a = try sc.res(mlx.mlx_to_fp8(&codes, normalized, sc.s), &codes);
            var decoded = mlx.mlx_array_new();
            const decoded_a = try sc.res(mlx.mlx_from_fp8(&decoded, codes_a, .float32, sc.s), &decoded);
            emb = try sc.mul(decoded_a, scale);
        }
        var parts = try self.a.alloc(A, nh);
        defer self.a.free(parts);
        for (0..nh) |h| {
            const one = try sc.slice(emb, -2, h, h + 1);
            parts[h] = try sc.rms(try sc.reshape(one, &.{ ci(T), ci(e.head_dim) }), e.per_head_norm[h], cfg.eps);
        }
        const flat = try sc.reshape(try sc.concat(parts, -1), &.{ ci(T), ci(nh * e.head_dim) });
        var vproj = try lin(sc, flat, e.value_w);
        if (e.value_bias.ctx != null) vproj = try sc.add(vproj, try sc.astype(e.value_bias, .float32));
        const value = try sc.mul(vproj, try sc.rmsUnweighted(vproj, cfg.eps));
        const main = try sc.mul(value, try sc.astype(e.pre_gamma, .float32));
        if (li == 1 or li == 21) { try sc.di(li, "engram_value", value); try sc.di(li, "engram_main", main); }
        var gates = try self.a.alloc(A, cfg.hc_mult);
        defer self.a.free(gates);
        for (0..cfg.hc_mult) |h| {
            var key_proj = try lin(sc, flat, e.key_w[h]);
            if (e.key_bias[h].ctx != null) key_proj = try sc.add(key_proj, try sc.astype(e.key_bias[h], .float32));
            const key = try sc.rms(key_proj, e.key_norm[h], cfg.eps);
            const qh = try sc.reshape(try sc.slice(streams, 1, h, h + 1), &.{ ci(T), ci(cfg.hidden) });
            const query = try sc.rms(qh, e.query_norm[h], cfg.eps);
            const dot = try sc.sum(try sc.mul(key, query), -1, true);
            const scaled = try sc.mul(dot, try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden)))));
            const root = try sc.sqrt(try sc.maximum(try sc.abs(scaled), try sc.f(1e-6)));
            gates[h] = try sc.sigmoid(try sc.mul(try sc.sign(scaled), root));
        }
        var branches = try self.a.alloc(A, cfg.hc_mult);
        defer self.a.free(branches);
        for (0..cfg.hc_mult) |h| branches[h] = try sc.mul(main, try sc.astype(e.scales[h], .float32));
        const gate_stack = try sc.stack(gates, 1); // [T, hc, 1]
        if (li == 1 or li == 21) try sc.di(li, "engram_gates", gate_stack);
        const en_out = try sc.add(try sc.mul(gate_stack, try sc.expand(main, 1)), try sc.stack(branches, 1));
        if (li == 1 or li == 21) try sc.di(li, "engram_out", en_out);
        return en_out;
    }

    pub fn resetState(self: *Model) void {
        self.state.reset();
        @memset(self.state.hist, @intCast(self.cfg.pad));
    }

    /// Prefill/extend: consumes `ids`, returns logits for every token [T, V].
    pub fn forward(self: *Model, ids: []const u32) !A {
        const st = &self.state;
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const T = ids.len;
        const id_i32 = try self.a.alloc(i32, T);
        defer self.a.free(id_i32);
        for (ids, 0..) |id, i| id_i32[i] = @intCast(id);
        const embIds = try sc.ints(id_i32, &.{ci(T)});
        try sc.d("ids", embIds);
        const embedF = try self.embed.rows(&sc, embIds, .float32);
        try sc.d("emb", embedF);
        var streams = try sc.broadcast(try sc.expand(embedF, 1), &.{ ci(T), ci(self.cfg.hc_mult), ci(self.cfg.hidden) });
        for (0..self.cfg.layers) |li| {
            streams = try self.layer(&sc, li, streams, st, ids);
        }
        const flat = try sc.reshape(streams, &.{ ci(T), ci(self.cfg.hc_mult * self.cfg.hidden) });
        const inv = try sc.rmsUnweighted(flat, self.cfg.eps);
        const mixes = try sc.mul(try lin(&sc, flat, self.hc_fn), inv);
        const pre = try sc.sigmoid(try sc.add(try sc.mul(mixes, try sc.astype(self.hc_scale, .float32)), try sc.astype(self.hc_base, .float32)));
        const pre_e = try sc.add(pre, try sc.f(self.cfg.hc_eps));
        const collapsed = try sc.sum(try sc.mul(try sc.expand(pre_e, 2), streams), 1, false);
        const normed = try sc.rms(collapsed, self.norm, self.cfg.eps);
        // All positions' logits [T, V]; the caller takes the last row.
        const logits = try lin(&sc, normed, self.head);
        try sc.d("final_logits", logits);
        st.len += T;
        // Carry the tail of this chunk for the next chunk's n-grams. hist[0]
        // is the token immediately before the next chunk, oldest first.
        if (T >= 2) {
            st.hist[0] = ids[T - 1]; // t-1 of the next chunk
            st.hist[1] = ids[T - 2]; // t-2
        } else if (T == 1) {
            var k: usize = st.hist.len - 1;
            while (k > 0) : (k -= 1) st.hist[k] = st.hist[k - 1];
            st.hist[0] = ids[0];
        }
        return sc.out(logits);
    }

    fn layer(self: *Model, sc: *Scope, li: usize, streams_in: A, st: *State, ids: []const u32) !A {
        var streams = streams_in;
        if (self.engramPresent(li)) {
            const en_out = try self.engram(sc, li, streams, ids, st);
            streams = try sc.add(streams, en_out);
        }
        var q_resid: ?A = null;
        const hc1 = try hci(sc, streams, self.layers[li].hc_attn_fn, self.layers[li].hc_attn_base, self.layers[li].hc_attn_scale, self.cfg);
        const normed1 = try sc.rms(hc1.out, self.layers[li].attn_norm, self.cfg.eps);
        if (li <= 1 or li == 21) try sc.di(li, "input_layernorm", normed1);
        const att = try self.attention(sc, li, normed1, &q_resid, st, st.len);
        const post1 = try sc.expand(try sc.astype(hc1.post, .float32), 2);
        streams = try sc.astype(try sc.add(try sc.mul(post1, try sc.expand(att, 1)), streams), .float32);
        if (li <= 1 or li == 21) try sc.di(li, "after_attn_residual", streams);

        const hc2 = try hci(sc, streams, self.layers[li].hc_ffn_fn, self.layers[li].hc_ffn_base, self.layers[li].hc_ffn_scale, self.cfg);
        const normed2 = try sc.rms(hc2.out, self.layers[li].ffn_norm, self.cfg.eps);
        if (li <= 1 or li == 21) try sc.di(li, "post_attention_layernorm", normed2);
        const ffn = try self.mlp(sc, li, normed2);
        const post2 = try sc.expand(try sc.astype(hc2.post, .float32), 2);
        streams = try sc.add(try sc.mul(post2, try sc.expand(ffn, 1)), streams);
        if (li <= 1 or li == 21) try sc.di(li, "after_ffn_residual", streams);
        return streams;
    }

    fn engramPresent(self: *const Model, li: usize) bool {
        for (self.engrams) |e| if (e.li == li) return true;
        return false;
    }
};

fn sizesPerLayer(all: []const []const usize, k: usize) []const usize {
    if (all.len == 0) return &.{};
    return all[@min(k, all.len - 1)];
}

fn hashMultipliers(layer_id: usize) []const i64 {
    return switch (layer_id) {
        2 => &.{ 62749575316035, 30426722171175, 18574837598557 },
        12 => &.{ 14916611332703, 34290019069965, 64992498366675 },
        22 => &.{ 55942601167937, 16505523750331, 42928537650479 },
        32 => &.{ 22330137526737, 12836963598171, 17229190754683 },
        else => &.{ 1, 3, 5 },
    };
}

fn sliceEvery(sc: *Scope, x: A, stride: usize, n: usize) !A {
    var idx = try sc.a.alloc(i32, n);
    defer sc.a.free(idx);
    for (0..n) |i| idx[i] = @intCast(i * stride);
    const idA = try sc.ints(idx, &.{ci(n)});
    return sc.take(x, idA, 0);
}

/// Reference's `apply_rotary_pos_emb`: cos/sin are already in the
/// `repeat_interleave(2)` layout, and `rotate_half` swaps even/odd pairs.
fn ropeRotate(sc: *Scope, x: A, cos: A, sin: A) !A {
    const sh = mlx.getShape(x);
    const rope_dim = sh[sh.len - 1];
    const half = @divExact(rope_dim, 2);
    var pair_shape: [4]c_int = undefined;
    for (0..sh.len - 1) |i| pair_shape[i] = sh[i];
    pair_shape[sh.len - 1] = half;
    pair_shape[sh.len] = 2;
    const xr = try sc.reshape(x, pair_shape[0 .. sh.len + 1]);
    const even = try sc.slice(xr, -1, 0, 1);
    const odd = try sc.slice(xr, -1, 1, 2);
    const turned = try sc.reshape(
        try sc.stack(&.{ try sc.op1(mlx.mlx_negative, odd), even }, -1),
        sh,
    );
    const c = if (sh.len == 3) try sc.expand(cos, 1) else cos;
    const s = if (sh.len == 3) try sc.expand(sin, 1) else sin;
    return sc.add(try sc.mul(x, c), try sc.mul(turned, s));
}

fn partialRope(sc: *Scope, x: A, cos: A, sin: A, rope_dim: usize) !A {
    const past = try sc.slice(x, -1, 0, dim(x, -1) - rope_dim);
    const rot = try sc.slice(x, -1, dim(x, -1) - rope_dim, dim(x, -1));
    const q = try ropeRotate(sc, rot, cos, sin);
    return sc.concat(&.{ past, q }, -1);
}

fn topkIdx(sc: *Scope, x: A, k: usize) !A {
    // x: [T, N]; return [T, k] of the largest indices (ties: lower first).
    // Host selection is deterministic and avoids argpartition's unstable
    // ordering; T is one at decode and a chunk at prefill, N is the expert or
    // group count, so the O(N*k) pass is small next to the expert GEMMs.
    const sh = mlx.getShape(x);
    const rows: usize = @intCast(sh[0]);
    const cols: usize = @intCast(sh[1]);
    const hv = try hostF32A(sc.a, x);
    defer sc.a.free(hv);
    const idx = try sc.a.alloc(i32, rows * k);
    defer sc.a.free(idx);
    const kk = @min(k, cols);
    for (0..rows) |r| {
        const row = hv[r * cols ..][0..cols];
        const out = idx[r * k ..][0..k];
        for (0..kk) |j| out[j] = -1;
        for (0..cols) |c| {
            var pos: usize = kk;
            for (0..kk) |j| {
                const cur = out[j];
                if (cur == -1 or row[c] > row[@intCast(cur)] or (row[c] == row[@intCast(cur)] and c < @as(usize, @intCast(cur)))) {
                    pos = j;
                    break;
                }
            }
            if (pos < kk) {
                var j = kk - 1;
                while (j > pos) : (j -= 1) out[j] = out[j - 1];
                out[pos] = @intCast(c);
            }
        }
    }
    return sc.ints(idx, &.{ ci(rows), ci(k) });
}

fn gatherGu(sc: *Scope, bank: QLin, latent: A, idx: A, cfg: Config) !A {
    // bank [E, 2*I, L], latent [T, L], idx [T, k] -> [T, k, 2*I].
    if (bank.s.ctx != null) {
        const uidx = try sc.astype(idx, .uint32);
        const xe = try sc.reshape(latent, &.{ ci(dim(latent, 0)), ci(1), ci(1), ci(cfg.latent) });
        if (dbgOn() and dbg_gu < 2) {
            dbg_gu += 1;
            dbgShape("gu.idx", idx);
            dbgShape("gu.x", xe);
            dbgShape("gu.w", bank.w);
            std.debug.print("[hy4dbg] gu quantized={} bits={} gs={}\n", .{ bank.s.ctx != null, bank.qp.bits, bank.qp.group_size });
        }
        const y = try gatherMm(sc, xe, &bank, uidx);
        if (dbgOn() and dbg_gu <= 2) dbgShape("gu.y", y);
        return sc.reshape(y, &.{ ci(dim(latent, 0)), ci(dim(idx, -1)), ci(2 * cfg.moe_inter) });
    }
    const T = dim(latent, 0);
    const k = dim(idx, -1);
    const sel = try sc.take(bank.w, try sc.astype(idx, .int32), 0);
    const le = try sc.reshape(try sc.expand(latent, 1), &.{ ci(T), ci(1), ci(1), ci(cfg.latent) });
    const prod = try sc.mul(sel, try sc.broadcast(le, &.{ ci(T), ci(k), ci(2 * cfg.moe_inter), ci(cfg.latent) }));
    return sc.sum(prod, -1, false);
}

fn gatherDown(sc: *Scope, bank: QLin, act: A, idx: A, cfg: Config) !A {
    // bank [E, L, I], act [T, k, I] -> [T, k, L].
    if (bank.s.ctx != null) {
        const uidx = try sc.astype(idx, .uint32);
        const T = dim(act, 0);
        const k = dim(idx, -1);
        // gather_qmm's output shape is lhs_idx.shape ++ [x.shape[-2], out];
        // act is [T,k,I], so insert the singleton row dim or it reads k as
        // the row dim and returns [T,k,k,L] (16x too many rows at top_k=16).
        const xe = try sc.reshape(act, &.{ ci(T), ci(k), ci(1), ci(cfg.moe_inter) });
        if (dbgOn() and dbg_dn < 2) {
            dbg_dn += 1;
            dbgShape("dn.idx", idx);
            dbgShape("dn.x", xe);
            dbgShape("dn.w", bank.w);
            std.debug.print("[hy4dbg] dn quantized={} bits={} gs={}\n", .{ bank.s.ctx != null, bank.qp.bits, bank.qp.group_size });
        }
        const y = try gatherMm(sc, xe, &bank, uidx);
        if (dbgOn() and dbg_dn <= 2) dbgShape("dn.y", y);
        return sc.reshape(y, &.{ ci(T), ci(k), ci(cfg.latent) });
    }
    const T = dim(act, 0);
    const k = dim(idx, -1);
    const sel = try sc.take(bank.w, try sc.astype(idx, .int32), 0);
    const ae = try sc.reshape(try sc.expand(act, 1), &.{ ci(T), ci(k), ci(1), ci(cfg.moe_inter) });
    const prod = try sc.mul(sel, try sc.broadcast(ae, &.{ ci(T), ci(k), ci(cfg.latent), ci(cfg.moe_inter) }));
    return sc.sum(prod, -1, false);
}

// ── tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

fn hostF32A(a: std.mem.Allocator, arr: A) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, arr, false, mlx.gpuStream()));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return a.dupe(f32, d[0..mlx.mlx_array_size(f)]);
}

fn hostF32(a: A) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, mlx.gpuStream()));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return testing.allocator.dupe(f32, d[0..mlx.mlx_array_size(f)]);
}

fn loadWeightsFile(a: std.mem.Allocator, dir: []const u8, filename: []const u8) !Weights {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, filename });
    defer a.free(path);
    return weights_load.loadWeightsSingleFile(a, path);
}

test "hy_v4_flash fixture: logits match the NumPy reference (HY4_TINY)" {
    const root = std.mem.span(std.c.getenv("HY4_TINY") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    const a = testing.allocator;
    const cfg_path = try std.fmt.allocPrint(a, "{s}/config.json", .{root});
    defer a.free(cfg_path);
    const cfg_json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 20));
    defer a.free(cfg_json);
    const cfg = try parseConfigJson(a, cfg_json);
    defer a.free(cfg.indexed);
    defer a.free(cfg.sparse);
    defer a.free(cfg.engram_layers);
    defer a.free(cfg.engram_sizes);
    defer a.free(cfg.engram_sizes[0]);
    defer a.free(cfg.hash_mult);
    var w = try loadWeightsFile(a, root, "model.safetensors");
    defer w.deinit();
    var fx = try loadWeightsFile(a, root, "fixture.safetensors");
    defer fx.deinit();
    var m = try Model.init(a, cfg, &w, mlx.gpuStream());
    defer m.deinit();

    const idsA = fx.get("ids") orelse return error.MissingFixture;
    try mlx.check(mlx.mlx_array_eval(idsA));
    const idp = mlx.mlx_array_data_int32(idsA) orelse return error.NoData;
    const ids = try a.alloc(u32, mlx.mlx_array_size(idsA));
    defer a.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(idp[i]);

    const gotA = try m.forward(ids);
    defer _ = mlx.mlx_array_free(gotA);
    const got = try hostF32(gotA);
    defer a.free(got);
    const want = try hostF32(fx.get("logits") orelse return error.MissingFixture);
    defer a.free(want);
    try testing.expectEqual(want.len, got.len);
    var worst: f64 = 0;
    for (got, want) |g, wa| worst = @max(worst, @abs(@as(f64, g) - wa));
    std.debug.print("[hy4] max |logit diff| = {e:.3}\n", .{worst});
    try testing.expect(worst < 2e-2);
}

test "hy_v4_flash fixture: quantized affine + external engram matches the NumPy reference (HY4_QUANT)" {
    const root = std.mem.span(std.c.getenv("HY4_QUANT") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    const a = testing.allocator;
    const cfg_path = try std.fmt.allocPrint(a, "{s}/config.json", .{root});
    defer a.free(cfg_path);
    const cfg_json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 20));
    defer a.free(cfg_json);
    var cfg = try parseConfigJson(a, cfg_json);
    defer a.free(cfg.indexed);
    defer a.free(cfg.sparse);
    defer a.free(cfg.engram_layers);
    defer a.free(cfg.engram_sizes);
    defer a.free(cfg.engram_sizes[0]);
    defer a.free(cfg.hash_mult);
    defer if (cfg.engram_table) |sp| {
        a.free(sp.file);
        a.free(sp.layers);
        a.free(sp.table_rows);
    };
    cfg.model_dir = root;
    try testing.expect(cfg.engram_table != null);
    var w = try loadWeightsFile(a, root, "model.safetensors");
    defer w.deinit();
    var fx = try loadWeightsFile(a, root, "fixture.safetensors");
    defer fx.deinit();
    var m = try Model.init(a, cfg, &w, mlx.gpuStream());
    defer m.deinit();
    try testing.expect(m.ext_table != null);

    const idsA = fx.get("ids") orelse return error.MissingFixture;
    try mlx.check(mlx.mlx_array_eval(idsA));
    const idp = mlx.mlx_array_data_int32(idsA) orelse return error.NoData;
    const ids = try a.alloc(u32, mlx.mlx_array_size(idsA));
    defer a.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(idp[i]);

    const gotA = try m.forward(ids);
    defer _ = mlx.mlx_array_free(gotA);
    const got = try hostF32(gotA);
    defer a.free(got);
    const want = try hostF32(fx.get("logits") orelse return error.MissingFixture);
    defer a.free(want);
    try testing.expectEqual(want.len, got.len);
    var worst: f64 = 0;
    for (got, want) |g, wa| worst = @max(worst, @abs(@as(f64, g) - wa));
    std.debug.print("[hy4-quant] max |logit diff| = {e:.3}\n", .{worst});
    try testing.expect(worst < 2e-2);

    // Regression: the same ids delivered in chunks, with t==0 at each chunk
    // boundary but real history behind it. The full-sequence logits above are
    // the oracle — the reference builds n-grams from the whole sequence with
    // position_ids, so a correct chunked path must reproduce every row.
    const vocab = mlx.getShape(gotA)[1];
    m.resetState();
    var from: usize = 0;
    var chunk_idx: usize = 0;
    while (from < ids.len) : (chunk_idx += 1) {
        const width: usize = if (from == 0) 4 else 1;
        const to = @min(from + width, ids.len);
        const chunkA = try m.forward(ids[from..to]);
        defer _ = mlx.mlx_array_free(chunkA);
        const chunk = try hostF32(chunkA);
        defer a.free(chunk);
        var cworst: f64 = 0;
        for (chunk, 0..) |g, j| {
            const wi = from * @as(usize, @intCast(vocab)) + j;
            cworst = @max(cworst, @abs(@as(f64, g) - @as(f64, want[wi])));
        }
        std.debug.print("[hy4-quant] chunk {d} ids[{d}..{d}] max |logit diff| = {e:.3}\n", .{ chunk_idx, from, to, cworst });
        try testing.expect(cworst < 2e-2);
        from = to;
    }
}

/// Offline gather correctness probe at the real pack's geometries. No pack,
/// no 70 GB: a random [E,out,in*bits/32] U32 bank is a valid affine bank.
///   HY4_GATHER_PROBE=1 zig build test -Dtest-filter="hy4 gather probe"
fn probeGather(a: std.mem.Allocator, label: []const u8, out_dim: c_int, bits: u32, gs: u32) !void {
    const s = mlx.gpuStream();
    const E: c_int = 512;
    const IN: c_int = 1536;
    const TOPK: usize = 16;
    const wc: c_int = @intCast(@divExact(@as(u32, @intCast(IN)) * bits, 32));
    const gc: c_int = @intCast(@divTrunc(@as(u32, @intCast(IN)), gs));
    std.debug.print("[probe] {s}: w=[{d},{d},{d}] bits={d} gs={d} wc={d} gc={d}\n", .{ label, E, out_dim, wc, bits, gs, wc, gc });

    // Any bit pattern is a valid quantized row, so build the U32 bank on the
    // host (mlx_random_bits only accepts widths 1/2/4, not 3).
    var prng = std.Random.DefaultPrng.init(20261119);
    const rnd = prng.random();
    const wbuf = try a.alloc(u32, @as(usize, @intCast(E)) * @as(usize, @intCast(out_dim)) * @as(usize, @intCast(wc)));
    defer a.free(wbuf);
    for (wbuf) |*v| v.* = rnd.int(u32);
    const wsh = [_]c_int{ E, out_dim, wc };
    const w = mlx.mlx_array_new_data(wbuf.ptr, &wsh, 3, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const gsh = [_]c_int{ E, out_dim, gc };
    const gn = @as(usize, @intCast(E)) * @as(usize, @intCast(out_dim)) * @as(usize, @intCast(gc));
    const sbuf = try a.alloc(u16, gn);
    defer a.free(sbuf);
    const bbuf = try a.alloc(u16, gn);
    defer a.free(bbuf);
    for (0..gn) |i| {
        const sbits: u32 = @bitCast(rnd.float(f32) * 0.05 + 0.01);
        const bbits: u32 = @bitCast(rnd.float(f32) - 0.5);
        sbuf[i] = @intCast(sbits >> 16);
        bbuf[i] = @intCast(bbits >> 16);
    }
    const sc = mlx.mlx_array_new_data(sbuf.ptr, &gsh, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(sc);
    const bi = mlx.mlx_array_new_data(bbuf.ptr, &gsh, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(bi);

    const xs = [_]c_int{ 1, 1, 1, IN };
    const xbuf = try a.alloc(f32, @intCast(IN));
    defer a.free(xbuf);
    for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
    const x = mlx.mlx_array_new_data(xbuf.ptr, &xs, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    var idx_i32: [TOPK]i32 = undefined;
    for (0..TOPK) |i| idx_i32[i] = @intCast((i * 37 + 5) % @as(usize, @intCast(E)));
    const ish = [_]c_int{ 1, @intCast(TOPK) };
    const idx = mlx.mlx_array_new_data(&idx_i32, &ish, 2, .int32);
    defer _ = mlx.mlx_array_free(idx);
    var uidx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uidx);
    try mlx.check(mlx.mlx_astype(&uidx, idx, .uint32, s));

    var got = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_gather_qmm(&got, x, w, sc, bi, mlx.mlx_array_new(), uidx, true,
        mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", false, s));

    // Reference: gather the SAME rows with take, dequantize with MLX, matmul.
    var wsel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsel);
    var ssel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ssel);
    var bsel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bsel);
    try mlx.check(mlx.mlx_take_axis(&wsel, w, idx, 0, s));
    try mlx.check(mlx.mlx_take_axis(&ssel, sc, idx, 0, s));
    try mlx.check(mlx.mlx_take_axis(&bsel, bi, idx, 0, s));
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, wsel, ssel, bsel,
        mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine",
        mlx.mlx_array_new(), .{ .value = .float32, .has_value = true }, s));
    const bshape = [_]c_int{ 1, @intCast(TOPK), 1, IN };
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_broadcast_to(&xb, x, &bshape, 4, s));
    var xt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xt);
    const ax = [_]c_int{ 0, 1, 3, 2 };
    try mlx.check(mlx.mlx_transpose_axes(&xt, xb, &ax, 4, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, deq, xt, s));

    const wstr = mlx.mlx_array_strides(w);
    std.debug.print("[probe] {s}: w.strides=[{d},{d},{d}] (row-contiguous would be [{d},{d},1])\n", .{
        label, wstr[0], wstr[1], wstr[2], @as(usize, @intCast(out_dim)) * @as(usize, @intCast(wc)), @as(usize, @intCast(wc)),
    });
    const gsh_got = mlx.getShape(got);
    const gsh_ref = mlx.getShape(ref);
    std.debug.print("[probe] got shape=[", .{});
    for (gsh_got, 0..) |d, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", d });
    std.debug.print("] ref shape=[", .{});
    for (gsh_ref, 0..) |d, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", d });
    std.debug.print("]\n", .{});

    const gv = try hostF32A(a, got);
    defer a.free(gv);
    const rv = try hostF32A(a, ref);
    defer a.free(rv);
    const n = @min(gv.len, rv.len);
    var worst: f64 = 0;
    var max_ref: f64 = 0;
    var bad: usize = 0;
    for (0..n) |i| {
        const d = @abs(@as(f64, gv[i]) - @as(f64, rv[i]));
        worst = @max(worst, d);
        max_ref = @max(max_ref, @abs(@as(f64, rv[i])));
    }
    const tol = max_ref * 0.01;
    for (0..n) |i| if (@abs(@as(f64, gv[i]) - @as(f64, rv[i])) > tol) {
        bad += 1;
    };
    std.debug.print("[probe] {s}: n={d} max|diff|={e:.4} max|ref|={e:.4} norm={e:.4} bad>{e:.4}={d}/{d} got[0]={e:.4} ref[0]={e:.4}\n", .{
        label, n, worst, max_ref, worst / @max(1e-8, max_ref), tol, bad, n, @as(f64, gv[0]), @as(f64, rv[0]),
    });

    // Bandwidth discriminator: same bank, 16 indices vs 1. If the kernel
    // reads the whole 512-expert bank regardless of idx, both times match.
    const empty = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty);
    const one = [_]i32{idx_i32[0]};
    const one_shape = [_]c_int{ 1, 1 };
    const idx1 = mlx.mlx_array_new_data(&one, &one_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(idx1);
    var uid1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uid1);
    try mlx.check(mlx.mlx_astype(&uid1, idx1, .uint32, s));
    const reps = 8;
    const t16 = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    for (0..reps) |_| {
        var g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&g, x, w, sc, bi, empty, uidx, true,
            mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", false, s));
        try mlx.check(mlx.mlx_array_eval(g));
        _ = mlx.mlx_array_free(g);
    }
    const ms16 = @as(f64, @floatFromInt(t16.read())) / 1e6 / @as(f64, @floatFromInt(reps));
    const t1 = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    for (0..reps) |_| {
        var g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&g, x, w, sc, bi, empty, uid1, true,
            mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", false, s));
        try mlx.check(mlx.mlx_array_eval(g));
        _ = mlx.mlx_array_free(g);
    }
    const ms1 = @as(f64, @floatFromInt(t1.read())) / 1e6 / @as(f64, @floatFromInt(reps));
    std.debug.print("[probe] {s}: gather ms 16idx={e:.3} 1idx={e:.3} ratio={e:.2}\n", .{ label, ms16, ms1, ms16 / @max(1e-9, ms1) });

    // Baseline: one expert via plain quantized matmul (reads ~1.8 MB).
    var w1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w1);
    var s1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s1);
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_take_axis(&w1, w, idx1, 0, s));
    try mlx.check(mlx.mlx_take_axis(&s1, sc, idx1, 0, s));
    try mlx.check(mlx.mlx_take_axis(&b1, bi, idx1, 0, s));
    const x2shape = [_]c_int{ 1, IN };
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &x2shape, 2, s));
    const tb = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    for (0..reps) |_| {
        var g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(&g, x2, w1, s1, b1, true,
            mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
        try mlx.check(mlx.mlx_array_eval(g));
        _ = mlx.mlx_array_free(g);
    }
    const ms1base = @as(f64, @floatFromInt(tb.read())) / 1e6 / @as(f64, @floatFromInt(reps));
    std.debug.print("[probe] {s}: 1-expert qmm baseline={e:.3} ms; gather1/baseline={e:.2}\n", .{ label, ms1base, ms1 / @max(1e-9, ms1base) });

    // Re-eval of an already-evaluated array: array::eval() only wait()s.
    try mlx.check(mlx.mlx_array_eval(got));
    const sw_re = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    for (0..100) |_| try mlx.check(mlx.mlx_array_eval(got));
    const us_re = @as(f64, @floatFromInt(sw_re.read())) / 100.0 / 1000.0;
    std.debug.print("[probe] {s}: re-eval of evaluated result = {e:.3} us/call\n", .{ label, us_re });
}

test "hy4 gather probe offline: gather_qmm vs dequant at pack shapes" {
    if (std.c.getenv("HY4_GATHER_PROBE") == null) return error.SkipZigTest;
    const a = testing.allocator;
    try probeGather(a, "gate_up 3bit", 3072, 3, 64);
    try probeGather(a, "down 4bit", 1536, 4, 64);
}

// Synthetic chain of 41 matmuls with (and without) a tiny host sync after
// each, mirroring the 53 router hostF32A reads + 4 Engram evals per token.
test "hy4 sync probe: price a host round-trip" {
    if (std.c.getenv("HY4_SYNC_PROBE") == null) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = mlx.gpuStream();
    const N: usize = 41;
    const D: c_int = 1024;
    const HALF: c_int = 512;

    var wbufs: [N][]f32 = undefined;
    var ws: [N]A = undefined;
    const wsh = [_]c_int{ D, D };
    for (0..N) |i| {
        wbufs[i] = try a.alloc(f32, @as(usize, @intCast(D)) * @as(usize, @intCast(D)));
        for (wbufs[i], 0..) |*v, j| v.* = @floatFromInt(@as(i32, @intCast((j + i) % 97)) - 48);
        ws[i] = mlx.mlx_array_new_data(wbufs[i].ptr, &wsh, 2, .float32);
    }
    const xbuf = try a.alloc(f32, @intCast(D));
    for (xbuf) |*v| v.* = 0.01;
    const xsh = [_]c_int{ 1, D };
    const x0 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 2, .float32);

    const io = std.Io.Threaded.global_single_threaded.io();

    // Warm up all kernels first so JIT cost does not bias one arm.
    {
        var warm = x0;
        for (0..N) |i| {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&o, warm, ws[i], s));
            warm = o;
        }
        try mlx.check(mlx.mlx_array_eval(warm));
    }

    // No intermediate sync: build the whole chain, one eval at the end.
    var sw = io_util.Stopwatch.init(io);
    var cur = x0;
    for (0..N) |i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&o, cur, ws[i], s));
        cur = o;
    }
    try mlx.check(mlx.mlx_array_eval(cur));
    const ms_plain = @as(f64, @floatFromInt(sw.read())) / 1e6;

    // With a tiny host sync after each matmul (what topkIdx does).
    sw = io_util.Stopwatch.init(io);
    cur = x0;
    var bytes_read: usize = 0;
    for (0..N) |i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&o, cur, ws[i], s));
        cur = o;
        const begin = [_]c_int{ 0, 0 };
        const end = [_]c_int{ 1, HALF };
        const stp = [_]c_int{ 1, 1 };
        var tiny = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&tiny, cur, &begin, 2, &end, 2, &stp, 2, s));
        try mlx.check(mlx.mlx_array_eval(tiny));
        if (mlx.mlx_array_data_float32(tiny)) |p| bytes_read += @intFromPtr(p) & 1;
    }
    const ms_sync = @as(f64, @floatFromInt(sw.read())) / 1e6;

    // Sync with a large pending op vs nothing pending.
    var big = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&big, cur, ws[0], s));
    sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(big));
    const us_after_big = @as(f64, @floatFromInt(sw.read())) / 1000.0;
    sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(cur));
    const us_idle = @as(f64, @floatFromInt(sw.read())) / 1000.0;

    std.debug.print("[syncprobe] N={d} D={d} plain={e:.3} ms  with={e:.3} ms  delta={e:.3} ms  per_sync={e:.4} ms (read={d})\n", .{
        N, D, ms_plain, ms_sync, ms_sync - ms_plain, (ms_sync - ms_plain) / @as(f64, @floatFromInt(N)), bytes_read,
    });
    std.debug.print("[syncprobe] eval with big op pending={e:.1} us  idle={e:.1} us\n", .{ us_after_big, us_idle });
}

// Host top-k vs a reference sort, on known synthetic router rows.
test "hy4 host topk audit: ties, negatives, indices and weights" {
    if (std.c.getenv("HY4_TOPK_AUDIT") == null) return error.SkipZigTest;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const s = mlx.gpuStream();
    const K: usize = 16;
    const C: c_int = 512;

    const cases = [_][]const u8{ "distinct", "ties", "negatives" };
    for (cases) |case| {
        const vals = try aa.alloc(f32, @intCast(C));
        for (0..@as(usize, @intCast(C))) |i| {
            if (std.mem.eql(u8, case, "distinct")) vals[i] = @floatFromInt(i)
            else if (std.mem.eql(u8, case, "ties")) vals[i] = @floatFromInt(i % 4)
            else vals[i] = -1.0e6 + @as(f32, @floatFromInt(i % 64));
        }
        const sh = [_]c_int{ 1, C };
        const arr = mlx.mlx_array_new_data(vals.ptr, &sh, 2, .float32);
        var sc = Scope.init(aa, s);
        defer sc.deinit();
        const got = try topkIdx(&sc, arr, K);
        try mlx.check(mlx.mlx_array_eval(got));
        const p = mlx.mlx_array_data_int32(got) orelse return error.NoData;
        const gotk = try a.alloc(i32, K);
        defer a.free(gotk);
        for (0..K) |i| gotk[i] = p[i];

        // Reference: sort by (value desc, index asc) — same tie rule as topkIdx.
        var refk: [K]usize = undefined;
        const c_usize: usize = @intCast(C);
        const used = try aa.alloc(bool, c_usize);
        @memset(used, false);
        for (0..K) |r| {
            var best: ?usize = null;
            for (0..c_usize) |i| {
                if (used[i]) continue;
                if (best == null or vals[i] > vals[best.?] or
                    (vals[i] == vals[best.?] and i < best.?)) best = i;
            }
            refk[r] = best.?;
            used[best.?] = true;
        }
        var ok = true;
        for (0..K) |i| if (gotk[i] != @as(i32, @intCast(refk[i]))) {
            ok = false;
            std.debug.print("[topkaudit] {s} MISMATCH at {d}: got {d} ref {d}\n", .{ case, i, gotk[i], refk[i] });
            break;
        };
        std.debug.print("[topkaudit] {s}: match={} got16={d},{d},{d},{d} ref16={d},{d},{d},{d}\n", .{
            case, ok, gotk[0], gotk[1], gotk[2], gotk[3], refk[0], refk[1], refk[2], refk[3],
        });
        try testing.expect(ok);
    }
}
