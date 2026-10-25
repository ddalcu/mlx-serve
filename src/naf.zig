//! NAF (valeoai/NAF, Apache-2.0): a vision-foundation-model feature upsampler, as Pixal3D reads it.
//!
//! The guide image runs through two small conv encoders (1x1 and reflect-padded 3x3, GroupNorm + SiLU, no
//! residual), concatenated to 256 channels, average-pooled to the output size and rotated by a 4-head 2-D
//! RoPE: those are the queries, and their pooling onto the low-res grid the keys. Each high-res pixel then
//! attends over the 9x9 low-res cells around its own (natten's dilated neighbourhood: the window shifts
//! inward at the borders, never pads), values the low-res features split into 4 heads.
//!
//! Pixal3D only reads the upsampled map bilinearly at projected points, so it is evaluated there and
//! nowhere else: a 1024 x 1024 x 1024 map is never built. Parity: `naf oracle` against the reference model
//! with natten's window (tests/dump_pixal3d_fixtures.py).

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const scope_mod = @import("mlx_scope.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = scope_mod.Scope;
const Weights = model_mod.Weights;

pub const DIM: u32 = 256;
const HEADS: u32 = 4;
const HEAD_DIM: u32 = DIM / HEADS;
const KERNEL: u32 = 9;
const GROUPS: c_int = 8;
const GN_EPS: f32 = 1e-5;
/// Tokens per graph: the value windows gathered for one chunk are 4 taps x 81 cells x C floats per token.
const CHUNK: usize = 256;

const Conv = struct { w: A, b: A, pad: c_int };
const EncBlock = struct { n1w: A, n1b: A, c1: Conv, n2w: A, n2b: A, c2: Conv };

const Encoder = struct {
    first: Conv,
    blocks: [2]EncBlock,
};

pub const Naf = struct {
    gpa: std.mem.Allocator,
    w: Weights,
    enc: Encoder,
    sem: Encoder,
    periods: [HEAD_DIM / 4]f32,
    owned: std.ArrayList(A) = .empty,

    /// `<dir>/model.safetensors`: the released state dict, keys unchanged.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8) !Naf {
        const w = try model_mod.loadWeightsFile(gpa, dir, "model.safetensors");
        var n = Naf{ .gpa = gpa, .w = w, .enc = undefined, .sem = undefined, .periods = undefined };
        errdefer n.deinit();
        n.enc = try n.encoder("image_encoder.encoder", 0);
        n.sem = try n.encoder("image_encoder.sem_encoder", 1);
        const p = try scope_mod.getW(&n.w, "image_encoder.rope.periods", .{});
        if (mlx.mlx_array_size(p) != n.periods.len) return error.BadNafCheckpoint;
        const s = mlx.gpuStream();
        defer _ = mlx.mlx_stream_free(s);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_astype(&c, p, .float32, s));
        try mlx.check(mlx.mlx_array_eval(c));
        @memcpy(&n.periods, (mlx.mlx_array_data_float32(c) orelse return error.MlxError)[0..n.periods.len]);
        return n;
    }

    pub fn deinit(self: *Naf) void {
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
        self.w.deinit();
    }

    fn encoder(self: *Naf, comptime prefix: []const u8, pad: c_int) !Encoder {
        var e: Encoder = undefined;
        e.first = try self.conv(prefix ++ ".0", pad);
        inline for (0..2) |i| {
            const b = prefix ++ std.fmt.comptimePrint(".{d}", .{i + 1});
            e.blocks[i] = .{
                .n1w = try scope_mod.getW(&self.w, b ++ ".norm1.weight", .{}),
                .n1b = try scope_mod.getW(&self.w, b ++ ".norm1.bias", .{}),
                .c1 = try self.conv(b ++ ".conv1", pad),
                .n2w = try scope_mod.getW(&self.w, b ++ ".norm2.weight", .{}),
                .n2b = try scope_mod.getW(&self.w, b ++ ".norm2.bias", .{}),
                .c2 = try self.conv(b ++ ".conv2", pad),
            };
        }
        return e;
    }

    /// Conv2d `[O, I, kh, kw]` -> channels-last `[O, kh, kw, I]` f32.
    fn conv(self: *Naf, comptime name: []const u8, pad: c_int) !Conv {
        const s = mlx.gpuStream();
        defer _ = mlx.mlx_stream_free(s);
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, try scope_mod.getW(&self.w, name ++ ".weight", .{}), &[_]c_int{ 0, 2, 3, 1 }, 4, s));
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, t, .float32, s));
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_array_eval(o));
        try self.owned.append(self.gpa, o);
        return .{ .w = o, .b = try scope_mod.getW(&self.w, name ++ ".bias", .{}), .pad = pad };
    }

    /// `image` `[1, H, W, 3]` f32 in [0, 1] (H == W, a multiple of `side`); the queries at `side` x `side`
    /// and their keys pooled onto the `lr` x `lr` grid of the features being upsampled.
    pub fn guide(self: *const Naf, alloc: std.mem.Allocator, s: S, image: A, side: u32, lr: u32) !Guide {
        const in_side: u32 = @intCast(mlx.getShape(image)[1]);
        if (in_side % side != 0 or side % lr != 0 or lr < KERNEL) return error.BadNafGeometry;
        var x: A = undefined;
        {
            const e = try runEncoder(alloc, s, &self.enc, image);
            defer _ = mlx.mlx_array_free(e);
            const m = try runEncoder(alloc, s, &self.sem, image);
            defer _ = mlx.mlx_array_free(m);
            var sc = Scope.init(alloc, s);
            defer sc.deinit();
            var cat = try sc.concat(&.{ e, m }, 3);
            if (in_side != side) cat = try blockMean(&sc, cat, in_side / side);
            const r = try self.rope(&sc, cat, side);
            try mlx.check(mlx.mlx_array_eval(r));
            x = sc.out(r);
        }
        errdefer _ = mlx.mlx_array_free(x);
        var sc = Scope.init(alloc, s);
        defer sc.deinit();
        const k = try blockMean(&sc, x, side / lr);
        const n_hr: c_int = @intCast(side * side);
        const n_lr: c_int = @intCast(lr * lr);
        const q = try sc.reshape(x, &.{ n_hr, DIM });
        const kk = try sc.reshape(k, &.{ n_lr, DIM });
        try mlx.check(mlx.mlx_array_eval(kk));
        const out = Guide{ .q = sc.out(q), .k = sc.out(kk), .side = side, .lr = lr };
        _ = mlx.mlx_array_free(x);
        return out;
    }

    /// 4-head axial RoPE over `[1, side, side, 256]`: rows then columns in [-1, 1], angles `2 pi c / period`,
    /// pair i with i + 32 inside each 64-wide head.
    fn rope(self: *const Naf, sc: *Scope, x: A, side: u32) !A {
        const n: usize = @as(usize, side) * side;
        const cos_t = try sc.a.alloc(f32, n * HEAD_DIM);
        defer sc.a.free(cos_t);
        const sin_t = try sc.a.alloc(f32, n * HEAD_DIM);
        defer sc.a.free(sin_t);
        const q = HEAD_DIM / 4;
        for (0..side) |i| for (0..side) |j| {
            const cy: f32 = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, @floatFromInt(side))) - 1.0;
            const cx: f32 = 2.0 * ((@as(f32, @floatFromInt(j)) + 0.5) / @as(f32, @floatFromInt(side))) - 1.0;
            const row = (i * side + j) * HEAD_DIM;
            for (0..q) |f| {
                const ay = 2.0 * std.math.pi * cy / self.periods[f];
                const ax = 2.0 * std.math.pi * cx / self.periods[f];
                for ([_]usize{ f, q + f }, [_]f32{ ay, ax }) |col, ang| {
                    cos_t[row + col] = @cos(ang);
                    sin_t[row + col] = @sin(ang);
                    cos_t[row + 2 * q + col] = @cos(ang);
                    sin_t[row + 2 * q + col] = @sin(ang);
                }
            }
        };
        const sd: c_int = @intCast(side);
        const tshape = [_]c_int{ 1, sd, sd, 1, HEAD_DIM };
        const cos = try sc.fromF32(cos_t, &tshape);
        const sin = try sc.fromF32(sin_t, &tshape);
        const h = try sc.reshape(x, &.{ 1, sd, sd, HEADS, HEAD_DIM });
        const half: c_int = HEAD_DIM / 2;
        const rot = try sc.concat(&.{ try sc.mulS(try sc.slice(h, 4, half, HEAD_DIM), -1), try sc.slice(h, 4, 0, half) }, 4);
        const y = try sc.add(try sc.mul(h, cos), try sc.mul(rot, sin));
        return sc.reshape(y, &.{ 1, sd, sd, DIM });
    }
};

/// The upsampler's query and key maps for one image, flattened row-major: `q` `[side^2, 256]`, `k` `[lr^2, 256]`.
pub const Guide = struct {
    q: A,
    k: A,
    side: u32,
    lr: u32,

    pub fn deinit(self: *Guide) void {
        _ = mlx.mlx_array_free(self.q);
        _ = mlx.mlx_array_free(self.k);
    }
};

/// One encoder over `image`, each conv in its own scope: at 1024 px every intermediate is half a gigabyte and a
/// scope keeps what it made until it ends. Caller frees.
fn runEncoder(alloc: std.mem.Allocator, s: S, e: *const Encoder, image: A) !A {
    var h = try layer(alloc, s, e.first, image, null);
    errdefer _ = mlx.mlx_array_free(h);
    for (e.blocks) |b| for ([_]Conv{ b.c1, b.c2 }, [_][2]A{ .{ b.n1w, b.n1b }, .{ b.n2w, b.n2b } }) |c, n| {
        const next = try layer(alloc, s, c, h, n);
        _ = mlx.mlx_array_free(h);
        h = next;
    };
    return h;
}

/// `conv(silu(groupNorm(x)))`, the norm skipped for an encoder's first conv; evaluated, caller frees.
fn layer(alloc: std.mem.Allocator, s: S, c: Conv, x: A, norm: ?[2]A) !A {
    var sc = Scope.init(alloc, s);
    defer sc.deinit();
    const in = if (norm) |n| try sc.silu(try groupNorm(&sc, x, n[0], n[1])) else x;
    const y = try applyConv(&sc, c, in);
    try mlx.check(mlx.mlx_array_eval(y));
    return sc.out(y);
}

fn applyConv(sc: *Scope, c: Conv, x: A) !A {
    const in = if (c.pad > 0) try reflectPad(sc, x, c.pad) else x;
    return sc.add(try sc.conv2d(in, c.w, 1, 0), try sc.astype(c.b, .float32));
}

/// Reflect padding (no edge repeat) of `[1, H, W, C]` on both spatial axes.
fn reflectPad(sc: *Scope, x: A, p: c_int) !A {
    var y = x;
    for ([_]usize{ 1, 2 }) |axis| {
        const n = mlx.getShape(y)[axis];
        var parts: [3]A = undefined;
        parts[1] = y;
        parts[0] = try flipAxis(sc, try sc.slice(y, axis, 1, 1 + p), @intCast(axis));
        parts[2] = try flipAxis(sc, try sc.slice(y, axis, n - 1 - p, n - 1), @intCast(axis));
        y = try sc.concat(&parts, @intCast(axis));
    }
    return y;
}

fn flipAxis(sc: *Scope, x: A, axis: c_int) !A {
    const n = mlx.getShape(x)[@intCast(axis)];
    if (n == 1) return x;
    const ids = try sc.a.alloc(i32, @intCast(n));
    defer sc.a.free(ids);
    for (ids, 0..) |*v, i| v.* = n - 1 - @as(i32, @intCast(i));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_take_axis(&o, x, try sc.fromI32(ids, &.{n}), axis, sc.s), &o);
}

/// `nn.GroupNorm(8, C)` on `[1, H, W, C]`: statistics per group over every position and its channels.
fn groupNorm(sc: *Scope, x: A, w: A, b: A) !A {
    const sh = mlx.getShape(x);
    const g = try sc.reshape(x, &.{ 1, sh[1], sh[2], GROUPS, @divExact(sh[3], GROUPS) });
    const mean = try nestedMean(sc, g);
    const d = try sc.sub(g, mean);
    var r = mlx.mlx_array_new();
    const inv = try sc.res(mlx.mlx_rsqrt(&r, try sc.addS(try nestedMean(sc, try sc.mul(d, d)), GN_EPS), sc.s), &r);
    const n = try sc.reshape(try sc.mul(d, inv), sh);
    return sc.add(try sc.mul(n, try sc.astype(w, .float32)), try sc.astype(b, .float32));
}

/// Mean over rows, columns and the group's channels of `[1, H, W, G, c]`, one axis at a time: a single f32
/// sum over a megapixel drifts by 1e-4, past what the reference's statistics allow.
fn nestedMean(sc: *Scope, g: A) !A {
    var y = g;
    for ([_]c_int{ 2, 1, 4 }) |axis| {
        var o = mlx.mlx_array_new();
        y = try sc.res(mlx.mlx_mean_axis(&o, y, axis, true, sc.s), &o);
    }
    return y;
}

/// Mean over `f` x `f` blocks of `[1, H, W, C]` (adaptive average pooling at an integer ratio).
fn blockMean(sc: *Scope, x: A, f: u32) !A {
    if (f == 1) return x;
    const sh = mlx.getShape(x);
    const fi: c_int = @intCast(f);
    const r = try sc.reshape(x, &.{ 1, @divExact(sh[1], fi), fi, @divExact(sh[2], fi), fi, sh[3] });
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_mean_axes(&o, r, &[_]c_int{ 2, 4 }, 2, false, sc.s), &o);
}

/// First low-res row (or column) of the window a high-res pixel at `i` attends: natten's neighbourhood
/// centre clamped into its dilation group of `lr` cells, the group being the pixel's own column of cells.
pub fn windowStart(i: u32, dilation: u32, lr: u32) u32 {
    const c = i / dilation;
    const half = KERNEL / 2;
    return @min(c -| half, lr - KERNEL);
}

/// The four bilinear taps `align_corners=False` with border padding takes at coordinate `p` on an
/// `r`-wide image when reading a `size`-wide map: indices and weights (the far tap clamps, at weight 0).
pub fn taps(p: f32, r: u32, size: u32) struct { lo: u32, hi: u32, w: f32 } {
    const sz: f32 = @floatFromInt(size);
    var u = (p + 0.5) * sz / @as(f32, @floatFromInt(r)) - 0.5;
    u = std.math.clamp(u, 0, sz - 1);
    const f = @floor(u);
    const lo: u32 = @intFromFloat(f);
    return .{ .lo = lo, .hi = @min(lo + 1, size - 1), .w = u - f };
}

/// The NAF-upsampled features at high-res pixels `pix` (`[n][2]` row, col): `[n, C]` f32. `vals` is the
/// low-res map `[lr^2, C]` (C a multiple of 4 heads).
pub fn atPixels(sc: *Scope, g: Guide, vals: A, pix: []const [2]u32) !A {
    const n = pix.len;
    const dil = g.side / g.lr;
    const kk: usize = KERNEL * KERNEL;
    const qi = try sc.a.alloc(i32, n);
    defer sc.a.free(qi);
    const ki = try sc.a.alloc(i32, n * kk);
    defer sc.a.free(ki);
    for (pix, 0..) |p, t| {
        qi[t] = @intCast(p[0] * g.side + p[1]);
        const sy = windowStart(p[0], dil, g.lr);
        const sx = windowStart(p[1], dil, g.lr);
        for (0..KERNEL) |a| for (0..KERNEL) |b| {
            ki[t * kk + a * KERNEL + b] = @intCast((sy + a) * g.lr + sx + b);
        };
    }
    const nc: c_int = @intCast(n);
    const c: c_int = mlx.getShape(vals)[1];
    const vh = @divExact(c, HEADS);
    const kidx = try sc.fromI32(ki, &.{@intCast(n * kk)});
    const q = try sc.reshape(try sc.take(g.q, try sc.fromI32(qi, &.{nc})), &.{ nc, HEADS, 1, HEAD_DIM });
    const k = try sc.transpose(try sc.reshape(try sc.take(g.k, kidx), &.{ nc, KERNEL * KERNEL, HEADS, HEAD_DIM }), &.{ 0, 2, 3, 1 });
    var sm = mlx.mlx_array_new();
    const p = try sc.res(mlx.mlx_softmax_axis(&sm, try sc.mulS(try sc.matmul(q, k), 1.0 / @sqrt(@as(f32, HEAD_DIM))), -1, true, sc.s), &sm);
    const v = try sc.transpose(try sc.reshape(try sc.take(vals, kidx), &.{ nc, KERNEL * KERNEL, HEADS, vh }), &.{ 0, 2, 1, 3 });
    return sc.reshape(try sc.matmul(p, v), &.{ nc, c });
}

/// Bilinear border-clamped reads at `pts` (`[n][2]` x, y pixel coordinates on an `r`-wide image, as the
/// projection lands them) of the low-res map `vals` (`[lr^2, C]`) and, with `g`, of its NAF upsampling:
/// `[n, C]` or `[n, 2C]` (low-res first) f32, evaluated chunk by chunk (caller frees).
pub fn samplePoints(alloc: std.mem.Allocator, s: S, pts: []const [2]f32, r: u32, vals: A, lr: u32, g: ?Guide) !A {
    var parts: std.ArrayList(A) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(alloc);
    }
    var start: usize = 0;
    while (start < pts.len) : (start += CHUNK) {
        const chunk = pts[start..@min(pts.len, start + CHUNK)];
        var sc = Scope.init(alloc, s);
        defer sc.deinit();
        var y = try bilinear(&sc, chunk, r, lr, vals, null);
        if (g) |gg| y = try sc.concat(&.{ y, try bilinear(&sc, chunk, r, gg.side, vals, gg) }, 1);
        try mlx.check(mlx.mlx_array_eval(y));
        try parts.append(alloc, sc.out(y));
    }
    var sc = Scope.init(alloc, s);
    defer sc.deinit();
    const all = if (parts.items.len == 1) parts.items[0] else try sc.concat(parts.items, 0);
    try mlx.check(mlx.mlx_array_eval(all));
    return sc.out(all);
}

/// One chunk: the four taps on a `size`-wide map, read from `vals` directly or through the upsampler.
fn bilinear(sc: *Scope, pts: []const [2]f32, r: u32, size: u32, vals: A, g: ?Guide) !A {
    const n = pts.len;
    const pix = try sc.a.alloc([2]u32, n * 4);
    defer sc.a.free(pix);
    const wt = try sc.a.alloc(f32, n * 4);
    defer sc.a.free(wt);
    for (pts, 0..) |p, t| {
        const tx = taps(p[0], r, size);
        const ty = taps(p[1], r, size);
        const rows = [2]u32{ ty.lo, ty.hi };
        const cols = [2]u32{ tx.lo, tx.hi };
        const wy = [2]f32{ 1 - ty.w, ty.w };
        const wx = [2]f32{ 1 - tx.w, tx.w };
        for (0..2) |a| for (0..2) |b| {
            pix[t * 4 + a * 2 + b] = .{ rows[a], cols[b] };
            wt[t * 4 + a * 2 + b] = wy[a] * wx[b];
        };
    }
    const nc: c_int = @intCast(n);
    const c: c_int = mlx.getShape(vals)[1];
    const read = if (g) |gg| try atPixels(sc, gg, vals, pix) else blk: {
        const ids = try sc.a.alloc(i32, n * 4);
        defer sc.a.free(ids);
        for (pix, ids) |p, *id| id.* = @intCast(p[0] * size + p[1]);
        break :blk try sc.take(vals, try sc.fromI32(ids, &.{nc * 4}));
    };
    const w = try sc.fromF32(wt, &.{ nc, 4, 1 });
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_sum_axis(&o, try sc.mul(try sc.reshape(read, &.{ nc, 4, c }), w), 1, false, sc.s), &o);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;
const t2 = @import("trellis2.zig");

test "naf window: natten's centre clamp keeps all 9 cells inside the group" {
    // 32 cells, dilation 16: pixel 0 and 63 open the grid, 16*16 sits mid-grid, the last pixel closes it.
    try testing.expectEqual(@as(u32, 0), windowStart(0, 16, 32));
    try testing.expectEqual(@as(u32, 0), windowStart(63, 16, 32));
    try testing.expectEqual(@as(u32, 12), windowStart(16 * 16, 16, 32));
    try testing.expectEqual(@as(u32, 23), windowStart(511, 16, 32));
    try testing.expectEqual(@as(u32, 23), windowStart(27 * 16, 16, 32)); // cell 27: centre clamps at 27
    try testing.expectEqual(@as(u32, 22), windowStart(26 * 16, 16, 32));
}

test "naf taps: border padding clamps the coordinate and the far tap carries no weight there" {
    const mid = taps(255.0, 512, 32); // (255.5) / 16 - 0.5 = 15.46875
    try testing.expectEqual(@as(u32, 15), mid.lo);
    try testing.expectEqual(@as(u32, 16), mid.hi);
    try testing.expectApproxEqAbs(@as(f32, 0.46875), mid.w, 1e-6);
    const lo = taps(-3.0, 512, 32);
    try testing.expectEqual(@as(u32, 0), lo.lo);
    try testing.expectEqual(@as(f32, 0), lo.w);
    const hi = taps(600.0, 512, 32);
    try testing.expectEqual(@as(u32, 31), hi.lo);
    try testing.expectEqual(@as(u32, 31), hi.hi);
    try testing.expectEqual(@as(f32, 0), hi.w);
}

test "naf group norm, reflect pad and block mean match a host reference" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    var x: [4 * 4 * 16]f32 = undefined;
    for (&x) |*v| v.* = rng.floatNorm(f32) * 2 + 1;
    var gw: [16]f32 = undefined;
    var gb: [16]f32 = undefined;
    for (&gw, &gb) |*a, *b| {
        a.* = rng.floatNorm(f32);
        b.* = rng.floatNorm(f32);
    }
    const xa = try sc.fromF32(&x, &.{ 1, 4, 4, 16 });
    const got = try t2.hostF32(&sc, try groupNorm(&sc, xa, try sc.fromF32(&gw, &.{16}), try sc.fromF32(&gb, &.{16})));
    for (0..8) |grp| {
        var m: f64 = 0;
        for (0..16) |p| for (0..2) |c| {
            m += x[p * 16 + grp * 2 + c];
        };
        m /= 32;
        var v: f64 = 0;
        for (0..16) |p| for (0..2) |c| {
            const d = x[p * 16 + grp * 2 + c] - m;
            v += d * d;
        };
        v /= 32;
        for (0..16) |p| for (0..2) |c| {
            const ch = grp * 2 + c;
            const want = (x[p * 16 + ch] - m) / @sqrt(v + GN_EPS) * gw[ch] + gb[ch];
            try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got[p * 16 + ch], 1e-4);
        };
    }
    // Reflect pad of a 3x3 single-channel ramp: row -1 mirrors row 1, never repeats row 0.
    const ramp = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const pad = try t2.hostF32(&sc, try reflectPad(&sc, try sc.fromF32(&ramp, &.{ 1, 3, 3, 1 }), 1));
    try testing.expectEqualSlices(f32, &.{ 4, 3, 4, 5, 4 }, pad[0..5]);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 1, 2, 1 }, pad[5..10]);
    const mean = try t2.hostF32(&sc, try blockMean(&sc, try sc.fromF32(&x, &.{ 1, 4, 4, 16 }), 2));
    try testing.expectApproxEqAbs((x[0] + x[16] + x[64] + x[80]) / 4, mean[0], 1e-5);
}

test "naf atPixels: each pixel attends its 9x9 window with per-head softmax over the low-res values" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const lr = 10;
    const side = 20;
    const c = 8; // 4 heads x 2 value dims
    const q = try gpa.alloc(f32, side * side * DIM);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, lr * lr * DIM);
    defer gpa.free(k);
    var v: [lr * lr * c]f32 = undefined;
    for (q) |*e| e.* = rng.floatNorm(f32) * 0.3;
    for (k) |*e| e.* = rng.floatNorm(f32) * 0.3;
    for (&v) |*e| e.* = rng.floatNorm(f32);
    const g = Guide{ .q = try sc.fromF32(q, &.{ side * side, DIM }), .k = try sc.fromF32(k, &.{ lr * lr, DIM }), .side = side, .lr = lr };
    const pix = [_][2]u32{ .{ 0, 0 }, .{ 19, 7 }, .{ 9, 12 }, .{ 3, 18 } };
    const got = try t2.hostF32(&sc, try atPixels(&sc, g, try sc.fromF32(&v, &.{ lr * lr, c }), &pix));
    for (pix, 0..) |p, t| {
        const sy = windowStart(p[0], 2, lr);
        const sx = windowStart(p[1], 2, lr);
        for (0..HEADS) |h| {
            var sc_: [81]f64 = undefined;
            var mx: f64 = -1e30;
            for (0..9) |a| for (0..9) |b| {
                var dot: f64 = 0;
                const cell = (sy + a) * lr + sx + b;
                for (0..HEAD_DIM) |d| dot += q[(p[0] * side + p[1]) * DIM + h * HEAD_DIM + d] * k[cell * DIM + h * HEAD_DIM + d];
                sc_[a * 9 + b] = dot / 8.0;
                mx = @max(mx, sc_[a * 9 + b]);
            };
            var sum: f64 = 0;
            for (&sc_) |*e| {
                e.* = @exp(e.* - mx);
                sum += e.*;
            }
            for (0..2) |d| {
                var o: f64 = 0;
                for (0..9) |a| for (0..9) |b| {
                    o += sc_[a * 9 + b] / sum * v[((sy + a) * lr + sx + b) * c + h * 2 + d];
                };
                try testing.expectApproxEqAbs(@as(f32, @floatCast(o)), got[t * c + h * 2 + d], 1e-4);
            }
        }
    }
}

test "naf oracle: the upsampled map at sampled pixels matches the reference (PIXAL3D_FIXTURES, PIXAL3D_NAF)" {
    const gpa = testing.allocator;
    const fixtures = std.mem.span(std.c.getenv("PIXAL3D_FIXTURES") orelse return error.SkipZigTest);
    const dir = std.mem.span(std.c.getenv("PIXAL3D_NAF") orelse return error.SkipZigTest);
    var naf = try Naf.open(gpa, dir);
    defer naf.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    for ([_][]const u8{ "naf_a.safetensors", "naf_b.safetensors" }) |file| {
        var fx = try model_mod.loadWeightsFile(gpa, fixtures, file);
        defer fx.deinit();
        var sc = Scope.init(gpa, s);
        defer sc.deinit();
        const image = try sc.transpose(fx.get("image").?, &.{ 0, 2, 3, 1 });
        const feats = fx.get("feats").?;
        const lr: u32 = @intCast(mlx.getShape(feats)[2]);
        const vals = try sc.reshape(try sc.transpose(feats, &.{ 0, 2, 3, 1 }), &.{ @intCast(lr * lr), 1024 });
        var g = try naf.guide(gpa, s, image, 512, lr);
        defer g.deinit();
        const pix_i = try t2.hostI32(gpa, fx.get("pix").?);
        defer gpa.free(pix_i);
        const pix = try gpa.alloc([2]u32, pix_i.len / 2);
        defer gpa.free(pix);
        for (pix, 0..) |*p, i| p.* = .{ @intCast(pix_i[i * 2]), @intCast(pix_i[i * 2 + 1]) };
        const got = try atPixels(&sc, g, vals, pix);
        try t2.expectClose(file, &sc, got, fx.get("out").?, 0.99999, 1e-4);
    }
}

/// A directory holding `model.safetensors` with ONE tensor that no loader reads, so every open fails after it owns the weights.
fn writeUselessCheckpoint(td: *std.testing.TmpDir) !void {
    const io = std.testing.io;
    const header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var buf: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memset(buf[8 + header.len ..], 0);
    try td.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &buf });
}

test "naf: a checkpoint missing its tensors is a named error and frees its weights once" {
    const io = std.testing.io;
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeUselessCheckpoint(&td);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try td.dir.realPath(io, &path_buf)];
    if (Naf.open(testing.allocator, dir)) |*n| {
        var naf = n.*;
        naf.deinit();
        return error.TestUnexpectedResult;
    } else |_| {}
}

