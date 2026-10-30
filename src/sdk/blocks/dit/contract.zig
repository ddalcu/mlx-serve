//! The dit area's kernel contract: the fused q/k/v head split with RMS norm and RoPE, the int8 attention pieces (row
//! and block quantizers, V prep, the NAX attention), and the w8a8 linear (rotate + quantize, the int8 GEMM with its
//! plain and SwiGLU epilogues). Listed in `kernel_contract/registry.zig`. Int8 outputs are checked by their stated
//! rule (`check`): a code is its value over the scale rounded to nearest, within the fp32 error of the division.
//! Heads and lengths are runtime arguments of these kernels (only the dtypes, the layout and the quantizer's block are
//! template arguments), so the served shapes keep a few heads: the float64 references fit a Debug build.
//! Known issues: the quantizers and the attention drop a NaN from a scale or a channel max (`metal::max`), so a NaN
//! input gives finite scales and outputs (gate 1: `scale_drops_nan`, `attn_drops_nan`); the NAX attention does not
//! build when kv heads x key blocks is under 8 (MLX binds `ks` in `constant`; gate 3: `attn_short_keys`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const dk = @import("dit_kernels.zig");
const ia = @import("int8_attn.zig");
const w8 = @import("w8a8.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

fn gamma(d: f64) f64 {
    return spec.gamma(d, 0x1p-24);
}

/// A code's allowance: rounding to nearest (half) plus the fp32 error of the quotient it rounds (`rel` of |t|) and
/// any absolute error `abs` of t.
fn codeOk(code: f64, t: f64, rel: f64, abs: f64) bool {
    const c = std.math.clamp(t, -127.0, 127.0);
    return @abs(code - c) <= 0.5 + rel * @abs(t) + abs;
}

fn fmtMsg(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch buf;
}

/// Non-finite plants at known rows of a [rows, n] input: NaN, +Inf, -Inf, both infinities in row 3.
fn plant(v: []f64, rows: usize, n: usize, cols: [4]usize) void {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const at = [_]struct { usize, usize, f64 }{ .{ 0, cols[0], nan }, .{ 1, cols[1], inf }, .{ 2, cols[2], -inf }, .{ 3, cols[3], inf }, .{ 3, cols[3] + 1, -inf } };
    for (at) |p| if (p[0] < rows) {
        v[p[0] * n + p[1]] = p[2];
    };
}

// ── q/k/v heads: RMS norm + RoPE ──

const Layout = struct { heads: usize, kv: usize, nt: usize, qo: usize, ko: usize, vo: usize, round: bool, eps: f64 };
/// Qwen-Image: 24 heads, q|k|v, the norm kept in f32.
const qwen_layout = Layout{ .heads = 24, .kv = 24, .nt = 3 * 3072, .qo = 0, .ko = 3072, .vo = 6144, .round = false, .eps = 1e-6 };
/// Krea: grouped k/v and the output gate's columns after v, the norm rounded to the stream dtype.
const krea_layout = Layout{ .heads = 24, .kv = 8, .nt = 2 * 3072 + 2 * 1024, .qo = 0, .ko = 3072, .vo = 4096, .round = true, .eps = 1e-5 };

const qkv_launch = metal.Launch{ .name = "msv_dit_qkv_heads", .inputs = &.{ "y", "qn", "kn", "cs", "sn", "eps" }, .outputs = &.{ "q", "k", "vo" }, .source = dk.QKV_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

const Qkv = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &qkv_launch,
    lay: Layout,
    dt: mlx.mlx_dtype = .bfloat16,
    shape: Shape = undefined,
    eps: f64 = 0,
    y: []f64 = &.{},
    qn: []f64 = &.{},
    kn: []f64 = &.{},
    cs: []f64 = &.{},
    sn: []f64 = &.{},
    args: [6]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, lay: Layout) Qkv {
        return .{ .rig = metal.Rig.init(a), .lay = lay };
    }

    pub fn deinit(self: *Qkv) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.y, self.qn, self.kn, self.cs, self.sn }) |v| a.free(v);
    }

    fn slots(self: *const Qkv) usize {
        return self.lay.heads + 2 * self.lay.kv;
    }

    pub fn geometry(self: *const Qkv, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0) return null;
        const l: c_int = @intCast(shape.rows);
        const sl: c_int = @intCast(self.slots());
        var g = metal.Geometry{ .grid = .{ 256 * @divTrunc(sl + 7, 8), l, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ 1, @intCast(self.lay.heads), l, 128 }, self.dt);
        g.out(&.{ 1, @intCast(self.lay.kv), l, 128 }, self.dt);
        g.out(&.{ 1, @intCast(self.lay.kv), l, 128 }, self.dt);
        g.dtype("T", self.dt);
        g.dtype("W", self.dt);
        g.int("H", self.lay.heads);
        g.int("KV", self.lay.kv);
        g.int("QO", self.lay.qo);
        g.int("KO", self.lay.ko);
        g.int("VO", self.lay.vo);
        g.flag("ROUND", self.lay.round);
        return g;
    }

    pub fn prepare(self: *Qkv, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const l = shape.rows;
        for ([_]*[]f64{ &self.y, &self.qn, &self.kn, &self.cs, &self.sn }, [_]usize{ l * self.lay.nt, 128, 128, l * 64, l * 64 }) |p, len| {
            a.free(p.*);
            p.* = &.{};
            p.* = try a.alloc(f64, len);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.eps = self.lay.eps;
        if (fill == .stress) {
            // Every value +-32: a head's sum of squares (131072) overflows fp16, 1 / rms is 1/32 exactly (eps 0) and
            // the rotation is a quarter turn, so every later stage is exact.
            self.eps = 0;
            for (self.y) |*v| v.* = if (rng.boolean()) 32 else -32;
            @memset(self.qn, 1);
            @memset(self.kn, 1);
            const turns = [4][2]f64{ .{ 1, 0 }, .{ 0, 1 }, .{ -1, 0 }, .{ 0, -1 } };
            for (self.cs, self.sn) |*c, *s| {
                const t = turns[rng.uintLessThan(usize, 4)];
                c.* = t[0];
                s.* = t[1];
            }
        } else {
            host.uniformInto(rng, self.y, -3, 3, self.dt);
            host.uniformInto(rng, self.qn, 0.5, 1.5, self.dt);
            host.uniformInto(rng, self.kn, 0.5, 1.5, self.dt);
            for (self.cs, self.sn) |*c, *s| {
                const th = rng.float(f64) * 2 * std.math.pi;
                c.* = spec.roundTo(.f32, @cos(th));
                s.* = spec.roundTo(.f32, @sin(th));
            }
            if (fill == .nonfinite) plant(self.y, l, self.lay.nt, .{ self.lay.qo + 5, self.lay.ko + 7, self.lay.vo + 3, self.lay.qo + 130 });
        }
        self.rig.reset();
        const li: c_int = @intCast(l);
        self.args[0] = try self.rig.input(self.dt, &.{ li, @intCast(self.lay.nt) }, self.y);
        self.args[1] = try self.rig.input(self.dt, &.{128}, self.qn);
        self.args[2] = try self.rig.input(self.dt, &.{128}, self.kn);
        self.args[3] = try self.rig.input(.float32, &.{ li, 64 }, self.cs);
        self.args[4] = try self.rig.input(.float32, &.{ li, 64 }, self.sn);
        self.args[5] = try self.rig.scalar(@floatCast(self.eps));
        return true;
    }

    pub fn run(self: *Qkv) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Qkv) usize {
        return self.slots() * self.shape.rows * 128;
    }

    fn rt(self: *const Qkv, v: f64) f64 {
        return host.rt(self.dt, v);
    }

    /// Every output, with each head's sum of squares in `acc` (null: float64); `flip` the spread a q/k output can
    /// take when 1 / rms moves within its fp32 error and the rounded norm lands on the other side.
    fn outputsIn(self: *const Qkv, want: []f64, mag: []f64, flip: ?[]f64, acc: ?spec.Accum, order: spec.Order) void {
        const l_n = self.shape.rows;
        const lay = self.lay;
        var sq: [128]f64 = undefined;
        var n: [3][128]f64 = undefined; // the norm at 1 / rms, and at its low and high ends
        const delta = gamma(128 + 6);
        for (0..self.slots()) |slot| for (0..l_n) |l| {
            const is_q = slot < lay.heads;
            const is_k = !is_q and slot < lay.heads + lay.kv;
            const head = if (is_q) slot else if (is_k) slot - lay.heads else slot - lay.heads - lay.kv;
            const col = (if (is_q) lay.qo else if (is_k) lay.ko else lay.vo) + head * 128;
            const base: usize = (if (is_q) 0 else if (is_k) lay.heads * l_n * 128 else (lay.heads + lay.kv) * l_n * 128) + (head * l_n + l) * 128;
            const src = self.y[l * lay.nt + col ..][0..128];
            if (!is_q and !is_k) {
                @memcpy(want[base..][0..128], src);
                @memset(mag[base..][0..128], 0);
                if (flip) |f| @memset(f[base..][0..128], 0);
                continue;
            }
            for (&sq, src) |*q, v| q.* = v * v;
            const ss = if (acc) |ac| spec.sumIn(ac, order, &sq) else host.sum(&sq);
            const inv = 1 / @sqrt(ss / 128 + self.eps);
            const w = if (is_q) self.qn else self.kn;
            for (0..3) |e| {
                const at = inv * (if (e == 1) 1 - delta else if (e == 2) 1 + delta else 1);
                for (&n[e], src, w) |*o, v, wi| {
                    const x = if (lay.round) self.rt(v * at) else v * at;
                    o.* = if (lay.round) self.rt(x * wi) else x * wi;
                }
            }
            for (0..64) |p| {
                const c = self.cs[l * 64 + p];
                const s = self.sn[l * 64 + p];
                const x = n[0][2 * p];
                const y = n[0][2 * p + 1];
                want[base + 2 * p] = x * c - y * s;
                want[base + 2 * p + 1] = y * c + x * s;
                mag[base + 2 * p] = @abs(x * c) + @abs(y * s);
                mag[base + 2 * p + 1] = @abs(y * c) + @abs(x * s);
                if (flip) |f| {
                    var dx: f64 = 0;
                    var dy: f64 = 0;
                    if (lay.round) for (1..3) |e| {
                        dx = @max(dx, @abs(n[e][2 * p] - x));
                        dy = @max(dy, @abs(n[e][2 * p + 1] - y));
                    };
                    f[base + 2 * p] = dx * @abs(c) + dy * @abs(s);
                    f[base + 2 * p + 1] = dy * @abs(c) + dx * @abs(s);
                }
            }
        };
    }

    pub fn reference(self: *const Qkv, want: []f64, mag: []f64) !void {
        self.outputsIn(want, mag, null, null, .sequential);
    }

    pub fn simulate(self: *const Qkv, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        self.outputsIn(out, mag, null, acc, order);
    }

    /// Gates 1 and 3: the bound, widened (ROUND) by a rounded norm landing one step over.
    pub fn check(self: *Qkv, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 4 * n);
        defer a.free(buf);
        const want = buf[0..n];
        const mag = buf[n .. 2 * n];
        const got = buf[2 * n .. 3 * n];
        const flip = buf[3 * n ..];
        self.outputsIn(want, mag, flip, null, .sequential);
        const s = qkvSpec("");
        for (got, 0..) |*g, i| g.* = spec.read(s.out, self.rig.outputBytes(), i);
        for (mag, want, flip) |*m, w, f| m.* = spec.bound(s, s.out, self.shape, w, m.*) + f;
        return host.within(&self.msg, at, got, want, mag);
    }

    fn wrapper(self: *Qkv) ![]const u8 {
        const x = self.args;
        const lay = self.lay;
        const h = try dk.qkvHeads(x[0], .{ .heads = @intCast(lay.heads), .kv_heads = @intCast(lay.kv), .q_off = @intCast(lay.qo), .k_off = @intCast(lay.ko), .v_off = @intCast(lay.vo) }, x[1], x[2], x[3], x[4], @floatCast(self.eps), lay.round, self.rig.stream());
        defer for ([_]mlx.mlx_array{ h.q, h.k, h.v }) |o| {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(&.{ h.q, h.k, h.v });
    }
};

fn qkvDepth(_: Shape) usize {
    return 128 + 6; // a head's sum of squares, then the mean, eps, rsqrt, the products and the rotation
}

fn qkvSpec(comptime name: []const u8) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = qkvDepth,
        .ops = 6,
        .out = .bf16,
        .served = &.{ .{ .rows = 256, .n = 1, .k = 128 }, .{ .rows = 77, .n = 1, .k = 128 } },
        // A sum of positive squares rounded in bf16 stays within the bf16 outputs' own rounding (as for the norms).
        .sums = .{ .bf16 = 0, .why = "positive squares: a bf16 sum stays within the bf16 outputs' rounding" },
    };
}

// ── int8 row and block quantizer ──

const Quant = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    rows: usize,
    d: usize,
    has_mean: bool,
    divide: bool,
    contig: bool,
    eps: f32,
    dt: mlx.mlx_dtype = .bfloat16,
    shape: Shape = undefined,
    /// x [1, heads, L, D] in head-major order, and the per-(head, channel) mean.
    x: []f64 = &.{},
    mean: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,
    /// The strided arm's input: a [1, L, heads, D] buffer seen through a transpose.
    view: ?mlx.mlx_array = null,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, rows: usize, d: usize, has_mean: bool, divide: bool, contig: bool, eps: f32) Quant {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .rows = rows, .d = d, .has_mean = has_mean, .divide = divide, .contig = contig, .eps = eps };
    }

    pub fn deinit(self: *Quant) void {
        const a = self.rig.a;
        self.dropView();
        self.rig.deinit();
        a.free(self.x);
        a.free(self.mean);
    }

    fn dropView(self: *Quant) void {
        if (self.view) |v| _ = mlx.mlx_array_free(v);
        self.view = null;
    }

    fn sg(self: *const Quant) c_int {
        return if (self.contig) ia.QUANT_SG else 1;
    }

    fn blocksPerHead(self: *const Quant, l: usize) usize {
        return (l + self.rows - 1) / self.rows;
    }

    pub fn geometry(self: *const Quant, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0) return null;
        const blocks: c_int = @intCast(shape.n * self.blocksPerHead(shape.rows));
        const sgv = self.sg();
        var g = metal.Geometry{ .grid = .{ @divTrunc(blocks + sgv - 1, sgv) * 32 * sgv, 1, 1 }, .tg = .{ 32 * sgv, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.n), @intCast(shape.rows), @intCast(self.d) }, .int8);
        g.out(&.{blocks}, .float32);
        g.int("SG", sgv);
        g.dtype("T", self.dt);
        g.int("D", self.d);
        g.int("ROWS", self.rows);
        g.flag("HAS_MEAN", self.has_mean);
        g.flag("DIVIDE", self.divide);
        g.flag("CONTIG", self.contig);
        return g;
    }

    pub fn prepare(self: *Quant, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const h = shape.n;
        const l = shape.rows;
        a.free(self.x);
        a.free(self.mean);
        self.x = &.{};
        self.mean = &.{};
        self.x = try a.alloc(f64, h * l * self.d);
        self.mean = try a.alloc(f64, h * self.d);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        host.uniformInto(rng, self.x, -4, 4, self.dt);
        // A per-channel offset, so the mean the K arm subtracts is not near zero.
        for (0..h) |hh| for (0..self.d) |c| {
            const off = host.rt(self.dt, rng.float(f64) * 4 - 2);
            for (0..l) |t| {
                const at = (hh * l + t) * self.d + c;
                self.x[at] = host.rt(self.dt, self.x[at] + off);
            }
        };
        for (0..h) |hh| for (0..self.d) |c| {
            var s: f64 = 0;
            for (0..l) |t| s += self.x[(hh * l + t) * self.d + c];
            self.mean[hh * self.d + c] = host.rt(self.dt, s / @as(f64, @floatFromInt(l)));
        };
        if (fill == .nonfinite) plant(self.x, h * l, self.d, .{ 3, 5, 7, 9 });
        try self.bind();
        return true;
    }

    fn bind(self: *Quant) !void {
        const rig = &self.rig;
        rig.reset();
        self.dropView();
        const h: c_int = @intCast(self.shape.n);
        const l: c_int = @intCast(self.shape.rows);
        const d: c_int = @intCast(self.d);
        if (self.contig) {
            self.args[0] = try rig.input(self.dt, &.{ 1, h, l, d }, self.x);
        } else {
            // Token-major in memory, read head-major through its strides (the NAX attention's q/k views).
            const tok = try rig.a.alloc(f64, self.x.len);
            defer rig.a.free(tok);
            for (0..self.shape.n) |hh| for (0..self.shape.rows) |t| {
                @memcpy(tok[(t * self.shape.n + hh) * self.d ..][0..self.d], self.x[(hh * self.shape.rows + t) * self.d ..][0..self.d]);
            };
            const raw = try rig.input(self.dt, &.{ 1, l, h, d }, tok);
            var v = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&v, raw, &[_]c_int{ 0, 2, 1, 3 }, 4, rig.stream()));
            self.view = v;
            self.args[0] = v;
        }
        self.args[1] = if (self.has_mean) try rig.input(self.dt, &.{ 1, h, d }, self.mean) else self.args[0];
        self.args[2] = try rig.scalar(self.eps);
    }

    pub fn run(self: *Quant) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Quant) usize {
        return self.x.len;
    }

    pub fn reference(self: *const Quant, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The rule: a block's scale is max(|x - mean|, eps) / 127 to within the division's fp32 error, NaN when the block
    /// holds one; every finite value's code is (x - mean) / scale (of the block's own written scale) rounded.
    pub fn check(self: *Quant, at: spec.Gate) !?[]const u8 {
        const out = self.rig.outputBytes();
        const n = self.x.len;
        const l = self.shape.rows;
        const nb = self.blocksPerHead(l);
        var planted: usize = 0;
        var finite: usize = 0;
        for (0..self.shape.n) |hh| for (0..nb) |b| {
            const r0 = b * self.rows;
            const nr = @min(self.rows, l - r0);
            var amax: f64 = 0;
            var nan = false;
            for (0..nr) |t| for (0..self.d) |c| {
                const v = self.value(hh, r0 + t, c);
                if (std.math.isNan(v)) nan = true;
                amax = @max(amax, @abs(v));
            };
            const want = if (nan) std.math.nan(f64) else @max(amax, self.eps) / 127;
            const got = spec.read(.f32, out[n..], hh * nb + b);
            if (spec.classOf(want) != .finite) planted += 1 else finite += 1;
            if (spec.classOf(got) != spec.classOf(want)) return fmtMsg(&self.msg, "block {d} of head {d}: scale {t} where float64 gives {t}", .{ b, hh, spec.classOf(got), spec.classOf(want) });
            if (spec.classOf(want) != .finite) continue;
            if (@abs(got - want) > 0x1p-22 * want) return fmtMsg(&self.msg, "block {d} of head {d}: scale {e} where float64 gives {e}", .{ b, hh, got, want });
            if (at != .shapes) continue;
            for (0..nr) |t| for (0..self.d) |c| {
                const i = (hh * l + r0 + t) * self.d + c;
                const code = host.intsOf(.int8, out, i);
                const v = self.value(hh, r0 + t, c);
                if (!codeOk(code, v / got, 0x1p-21, 0)) return fmtMsg(&self.msg, "value {d}: code {d} where {e} / {e} rounds to {d}", .{ i, code, v, got, @round(v / got) });
            };
        };
        if (at == .nonfinite and (planted == 0 or finite == 0)) return "the planted values leave no block of one class: no proof";
        return null;
    }

    /// x - mean as the kernel forms it (one fp32 subtraction of exact operands).
    fn value(self: *const Quant, hh: usize, t: usize, c: usize) f64 {
        const x: f32 = @floatCast(self.x[(hh * self.shape.rows + t) * self.d + c]);
        const m: f32 = if (self.has_mean) @floatCast(self.mean[hh * self.d + c]) else 0;
        return x - m;
    }

    fn wrapper(self: *Quant) ![]const u8 {
        const p = try ia.quantRows(self.args[0], @intCast(self.rows), if (self.has_mean) self.args[1] else null, self.eps, self.divide, self.contig, self.rig.stream());
        defer for (p) |x| {
            _ = mlx.mlx_array_free(x);
        };
        return self.rig.bytesOf(&p);
    }
};

fn quantLaunch(comptime name: [:0]const u8, comptime contig: bool, comptime m: u32) metal.Launch {
    return .{ .name = name, .inputs = &.{ "x", "mean", "eps" }, .outputs = &.{ "q", "scales" }, .source = ia.QUANT_SOURCE, .contiguous = contig, .tg_bytes = 0, .m1 = m, .m2 = m };
}
const quant_contig = quantLaunch("msv_int8_quant_rows_contig", true, 832);
const quant_strided = quantLaunch("msv_int8_quant_rows", false, 928);

fn zero(_: Shape) usize {
    return 0;
}

fn quantSpec(comptime name: []const u8, rows: usize, served: []const Shape, known: ?spec.Known) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = zero,
        .out = .f32,
        .nan = .{ .stated = "a block holding a NaN has a NaN scale; codes exist for finite values only" },
        .tiles = .{ .{ .size = rows }, .{}, .{} },
        .served = served,
        .known = known,
    };
}

// ── V prep ──

const vprep_launch = metal.Launch{ .name = "msv_i8attn_vprep", .inputs = &.{ "v", "vm" }, .outputs = &.{"vi"}, .source = ia.VPREP_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

const VPrep = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &vprep_launch,
    dt: mlx.mlx_dtype = .bfloat16,
    shape: Shape = undefined,
    v: []f64 = &.{},
    vm: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator) VPrep {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *VPrep) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.v);
        a.free(self.vm);
    }

    pub fn geometry(self: *const VPrep, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0) return null;
        const items: c_int = @intCast(shape.n * shape.rows);
        var g = metal.Geometry{ .grid = .{ 256 * @divTrunc(items + 7, 8), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ @intCast(shape.n), @intCast(shape.rows), 128 }, .int8);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *VPrep, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.v);
        a.free(self.vm);
        self.v = &.{};
        self.vm = &.{};
        self.v = try a.alloc(f64, shape.n * shape.rows * 128);
        self.vm = try a.alloc(f64, shape.n * 128);
        var prng = std.Random.DefaultPrng.init(seed);
        host.uniformInto(prng.random(), self.v, -4, 4, self.dt);
        if (fill == .nonfinite) plant(self.v, shape.n * shape.rows, 128, .{ 3, 5, 7, 9 });
        // The channel max |v| as the wrapper's MLX max forms it (a NaN stays NaN).
        for (0..shape.n) |h| for (0..128) |c| {
            var m: f64 = 0;
            for (0..shape.rows) |t| {
                const x = @abs(self.v[(h * shape.rows + t) * 128 + c]);
                m = if (std.math.isNan(x) or std.math.isNan(m)) std.math.nan(f64) else @max(m, x);
            }
            self.vm[h * 128 + c] = m;
        };
        self.rig.reset();
        self.args[0] = try self.rig.input(self.dt, &.{ 1, @intCast(shape.n), @intCast(shape.rows), 128 }, self.v);
        self.args[1] = try self.rig.input(.float32, &.{ 1, @intCast(shape.n), 128 }, self.vm);
        return true;
    }

    pub fn run(self: *VPrep) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const VPrep) usize {
        return self.v.len;
    }

    pub fn reference(_: *const VPrep, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The rule: where v and its channel max are finite, the code is v / (max(channel max, 1e-20) / 127) rounded.
    pub fn check(self: *VPrep, at: spec.Gate) !?[]const u8 {
        const out = self.rig.outputBytes();
        var skipped: usize = 0;
        for (self.v, 0..) |x, i| {
            const vm = self.vm[(i / (self.shape.rows * 128)) * 128 + i % 128];
            if (!std.math.isFinite(x) or !std.math.isFinite(vm)) {
                skipped += 1;
                continue;
            }
            const sc = @max(vm, 1e-20) / 127;
            const code = host.intsOf(.int8, out, i);
            if (!codeOk(code, x / sc, 0x1p-21, 0)) return fmtMsg(&self.msg, "value {d}: code {d} where {e} / {e} rounds to {d}", .{ i, code, x, sc, @round(x / sc) });
        }
        if (at == .nonfinite and skipped == 0) return "no non-finite value planted: no proof";
        return null;
    }

    fn wrapper(_: *VPrep) ![]const u8 {
        return error.CheckedThroughAttention;
    }
};

// ── int8 attention (NAX) ──

const attn_launch = metal.Launch{ .name = "msv_i8attn", .inputs = &.{ "qi", "qs", "ki", "ks", "vi", "vm", "scale" }, .outputs = &.{"o"}, .source = ia.ATTN_SOURCE, .header = ia.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };

/// Softmax(q k^T * scale) v over int8 codes: q [1, H, LQ, 128] with one scale a row, k [1, KV, L, 128] with one a
/// 32-key block, v [KV, L, 128] with one a channel (`vm` / 127). Shape: rows LQ, k L; heads fixed per registration.
const Attn = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &attn_launch,
    heads: usize,
    kv: usize,
    /// Gate 1 also plants a NaN channel max (the registration carrying that finding).
    nan_channel: bool,
    dt: mlx.mlx_dtype = .bfloat16,
    shape: Shape = undefined,
    scale: f64 = 0,
    qi: []f64 = &.{},
    qs: []f64 = &.{},
    ki: []f64 = &.{},
    ks: []f64 = &.{},
    vi: []f64 = &.{},
    vm: []f64 = &.{},
    args: [7]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    const BK = 32;

    fn init(a: std.mem.Allocator, heads: usize, kv: usize, nan_channel: bool) Attn {
        return .{ .rig = metal.Rig.init(a), .heads = heads, .kv = kv, .nan_channel = nan_channel };
    }

    pub fn deinit(self: *Attn) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.qi, self.qs, self.ki, self.ks, self.vi, self.vm }) |v| a.free(v);
    }

    fn nb(l: usize) usize {
        return (l + BK - 1) / BK;
    }

    pub fn geometry(self: *const Attn, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k < BK) return null;
        const lq: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ @divTrunc(lq + 63, 64) * 128, @intCast(self.heads), 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ lq, @intCast(self.heads * 128) }, self.dt);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *Attn, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        if (!w8.nax()) return error.NaxOnly;
        const a = self.rig.a;
        self.shape = shape;
        const lq = shape.rows;
        const l = shape.k;
        for ([_]*[]f64{ &self.qi, &self.qs, &self.ki, &self.ks, &self.vi, &self.vm }, [_]usize{ self.heads * lq * 128, self.heads * lq, self.kv * l * 128, self.kv * nb(l), self.kv * l * 128, self.kv * 128 }) |p, len| {
            a.free(p.*);
            p.* = &.{};
            p.* = try a.alloc(f64, len);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        host.intsInto(rng, self.qi, -127, 127);
        host.intsInto(rng, self.ki, -127, 127);
        host.intsInto(rng, self.vi, -127, 127);
        host.uniformInto(rng, self.qs, 0.001, 0.004, .float32);
        host.uniformInto(rng, self.ks, 0.001, 0.004, .float32);
        host.uniformInto(rng, self.vm, 0.5, 4, .float32);
        self.scale = spec.roundTo(.f32, 1.0 / @sqrt(128.0) * std.math.log2e);
        if (fill == .nonfinite) {
            // The float inputs: one row's q scale, one channel's max (Inf; NaN when asked).
            self.qs[0] = std.math.nan(f64);
            self.vm[9] = std.math.inf(f64);
            if (self.nan_channel) self.vm[5] = std.math.nan(f64);
        }
        self.rig.reset();
        const h: c_int = @intCast(self.heads);
        const kvh: c_int = @intCast(self.kv);
        self.args[0] = try self.rig.input(.int8, &.{ 1, h, @intCast(lq), 128 }, self.qi);
        self.args[1] = try self.rig.input(.float32, &.{@intCast(self.qs.len)}, self.qs);
        self.args[2] = try self.rig.input(.int8, &.{ 1, kvh, @intCast(l), 128 }, self.ki);
        self.args[3] = try self.rig.input(.float32, &.{@intCast(self.ks.len)}, self.ks);
        self.args[4] = try self.rig.input(.int8, &.{ kvh, @intCast(l), 128 }, self.vi);
        self.args[5] = try self.rig.input(.float32, &.{ 1, kvh, 128 }, self.vm);
        self.args[6] = try self.rig.scalar(@floatCast(self.scale));
        return true;
    }

    pub fn run(self: *Attn) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Attn) usize {
        return self.shape.rows * self.heads * 128;
    }

    /// Exact attention over the dequantized codes (base-2 logits: `scale` carries log2 e), and each output's
    /// allowance: P's int8 rounding moves an output by at most 0.5 / 127 of each block's weight times |v - o|, then
    /// fp32 and the output's rounding.
    fn outputs(self: *const Attn, want: []f64, allow: []f64) !void {
        const a = self.rig.a;
        const lq = self.shape.rows;
        const l = self.shape.k;
        const s = try a.alloc(f64, l);
        defer a.free(s);
        const wgt = try a.alloc(f64, l);
        defer a.free(wgt);
        const fb = try a.alloc(f64, nb(l));
        defer a.free(fb);
        for (0..self.heads) |h| {
            const kvh = h / (self.heads / self.kv);
            for (0..lq) |r| {
                const qrow = self.qi[(h * lq + r) * 128 ..][0..128];
                const c0 = self.qs[h * lq + r] * self.scale;
                var m = -std.math.inf(f64);
                for (s, 0..) |*o, j| {
                    const dot = host.dot(qrow, self.ki[(kvh * l + j) * 128 ..][0..128])[0];
                    o.* = dot * c0 * self.ks[kvh * nb(l) + j / BK];
                    m = @max(m, o.*);
                }
                // A block's weight relative to the row max: 2^(block max - row max).
                for (fb, 0..) |*f, b| {
                    var mb = -std.math.inf(f64);
                    for (b * BK..@min(l, b * BK + BK)) |j| mb = @max(mb, s[j]);
                    f.* = std.math.exp2(mb - m);
                }
                var den: f64 = 0;
                for (wgt, s) |*w, x| {
                    w.* = std.math.exp2(x - m);
                    den += w.*;
                }
                for (0..128) |c| {
                    const vc = self.vm[kvh * 128 + c] / 127;
                    var num: f64 = 0;
                    for (wgt, 0..) |w, j| num += w * self.vi[(kvh * l + j) * 128 + c];
                    const o = num * vc / den;
                    const i = (r * self.heads + h) * 128 + c;
                    want[i] = o;
                    var spread: f64 = 0;
                    var mag: f64 = 0;
                    for (0..l) |j| {
                        const v = self.vi[(kvh * l + j) * 128 + c] * vc;
                        spread += fb[j / BK] * @abs(v - o);
                        mag += wgt[j] * @abs(v);
                    }
                    // In P's units a block's max is 127, so the rounded denominator is at least 127 and at least the
                    // exact one less half a unit a key (each scaled by its block's weight); 2^-12 covers the fast exp2.
                    var units: f64 = 0;
                    for (0..l) |j| units += fb[j / BK];
                    const den_lo = @max(127.0, 127 * den - 0.5 * units);
                    const p_round = (0.5 + 0x1p-12) * spread / den_lo;
                    const work = p_round + gamma(@floatFromInt(nb(l) + 8)) * mag / den;
                    allow[i] = work + 0x1p-8 * (@abs(o) + work) + 0x1p-133;
                }
            }
        }
    }

    pub fn reference(self: *const Attn, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag);
    }

    /// The rule: within the allowance of `outputs`; a NaN or Inf scale or channel max reaches the outputs it touches.
    pub fn check(self: *Attn, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        try self.outputs(buf[0..n], buf[n .. 2 * n]);
        for (buf[2 * n ..], 0..) |*g, i| g.* = spec.read(.bf16, self.rig.outputBytes(), i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }

    /// The wrapper from float q, k, v; the same call through the prep wrappers and this launch.
    fn drift(self: *Attn, a: std.mem.Allocator) !struct { []u8, []const u8 } {
        const s = self.rig.stream();
        var prng = std.Random.DefaultPrng.init(5);
        const rng = prng.random();
        const lq = self.shape.rows;
        const l = self.shape.k;
        var vals = try a.alloc(f64, (self.heads * lq + 2 * self.kv * l) * 128);
        defer a.free(vals);
        host.uniformInto(rng, vals, -2, 2, .bfloat16);
        var scratch = metal.Rig.init(a);
        defer scratch.deinit();
        const q = try scratch.input(.bfloat16, &.{ 1, @intCast(self.heads), @intCast(lq), 128 }, vals[0 .. self.heads * lq * 128]);
        const k = try scratch.input(.bfloat16, &.{ 1, @intCast(self.kv), @intCast(l), 128 }, vals[self.heads * lq * 128 ..][0 .. self.kv * l * 128]);
        const v = try scratch.input(.bfloat16, &.{ 1, @intCast(self.kv), @intCast(l), 128 }, vals[(self.heads * lq + self.kv * l) * 128 ..]);
        const sc: f32 = 1.0 / @sqrt(128.0);
        const theirs_arr = (try ia.attention(q, k, v, sc, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(theirs_arr);
        const theirs = try a.dupe(u8, try scratch.bytesOf(&.{theirs_arr}));
        errdefer a.free(theirs);
        // Ours: the prep wrappers' outputs and V's channel max into the declared V prep and attention launches.
        const qp = try ia.prepQ(q, s);
        defer for (qp) |x| {
            _ = mlx.mlx_array_free(x);
        };
        const kp = try ia.prepK(k, s);
        defer for (kp) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var vp = VPrep.init(a);
        defer vp.deinit();
        vp.shape = .{ .rows = l, .n = self.kv, .k = 128 };
        const vhost = try host.read(a, v, s);
        defer a.free(vhost);
        vp.v = try a.dupe(f64, vhost);
        vp.vm = try a.alloc(f64, self.kv * 128);
        for (0..self.kv) |hh| for (0..128) |c| {
            var m: f64 = 0;
            for (0..l) |t| m = @max(m, @abs(vhost[(hh * l + t) * 128 + c]));
            vp.vm[hh * 128 + c] = m;
        };
        vp.args[0] = try vp.rig.input(.bfloat16, &.{ 1, @intCast(self.kv), @intCast(l), 128 }, vp.v);
        vp.args[1] = try vp.rig.input(.float32, &.{ 1, @intCast(self.kv), 128 }, vp.vm);
        try vp.run();
        const vi = try host.read(a, vp.rig.output(0), s);
        defer a.free(vi);
        for ([_]*[]f64{ &self.qi, &self.qs, &self.ki, &self.ks }, [_]mlx.mlx_array{ qp[0], qp[1], kp[0], kp[1] }) |dst, src| {
            a.free(dst.*);
            dst.* = &.{};
            dst.* = try host.read(a, src, s);
        }
        a.free(self.vi);
        self.vi = &.{};
        self.vi = try a.dupe(f64, vi);
        a.free(self.vm);
        self.vm = &.{};
        self.vm = try a.dupe(f64, vp.vm);
        self.scale = sc * std.math.log2e;
        self.rig.reset();
        const h: c_int = @intCast(self.heads);
        const kvh: c_int = @intCast(self.kv);
        self.args[0] = try self.rig.input(.int8, &.{ 1, h, @intCast(lq), 128 }, self.qi);
        self.args[1] = try self.rig.input(.float32, &.{@intCast(self.qs.len)}, self.qs);
        self.args[2] = try self.rig.input(.int8, &.{ 1, kvh, @intCast(l), 128 }, self.ki);
        self.args[3] = try self.rig.input(.float32, &.{@intCast(self.ks.len)}, self.ks);
        self.args[4] = try self.rig.input(.int8, &.{ kvh, @intCast(l), 128 }, self.vi);
        self.args[5] = try self.rig.input(.float32, &.{ 1, kvh, 128 }, self.vm);
        self.args[6] = try self.rig.scalar(@floatCast(self.scale));
        try self.run();
        return .{ try a.dupe(u8, self.rig.outputBytes()), theirs };
    }
};

fn attnSpec(comptime name: []const u8, known: ?spec.Known) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = zero,
        .out = .bf16,
        .nan = .{ .stated = "a NaN or Inf q scale, key-block scale or channel max reaches every output it touches" },
        // 64 query rows a threadgroup; keys in 32-key blocks (at least one).
        .tiles = .{ .{ .size = 64 }, .{}, .{ .size = 32, .min = 32 } },
        .served = &.{ .{ .rows = 100, .n = 1, .k = 300 }, .{ .rows = 64, .n = 1, .k = 256 } },
        .known = known,
    };
}

// ── w8a8: rotate + quantize ──

const rq_launch = metal.Launch{ .name = "msv_w8a8_rq", .inputs = &.{ "x", "signs" }, .outputs = &.{ "xq", "xs" }, .source = w8.RQ_SOURCE, .tg_bytes = 32, .m1 = 512, .m2 = 512 };

const Rq = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &rq_launch,
    dt: mlx.mlx_dtype = .bfloat16,
    shape: Shape = undefined,
    x: []f64 = &.{},
    signs: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator) Rq {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *Rq) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.x);
        a.free(self.signs);
    }

    pub fn geometry(self: *const Rq, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k == 0 or shape.k % 256 != 0) return null;
        var g = metal.Geometry{ .grid = .{ 256, @intCast(shape.rows), 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ @intCast(shape.rows), @intCast(shape.k) }, .int8);
        g.out(&.{@intCast(shape.rows)}, .float32);
        g.dtype("T", self.dt);
        g.int("CH", (shape.k / 256 + 7) / 8);
        return g;
    }

    pub fn prepare(self: *Rq, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.x);
        a.free(self.signs);
        self.x = &.{};
        self.signs = &.{};
        self.x = try a.alloc(f64, shape.rows * shape.k);
        // The served signs: every activation and weight of one width is rotated by the same vector.
        const signs = try w8.signsFor(@intCast(shape.k), self.dt, self.rig.stream());
        self.signs = try host.read(a, signs, self.rig.stream());
        var prng = std.Random.DefaultPrng.init(seed);
        host.uniformInto(prng.random(), self.x, -4, 4, self.dt);
        if (fill == .nonfinite) plant(self.x, shape.rows, shape.k, .{ 3, 300, 600, 900 });
        self.rig.reset();
        self.args[0] = try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.x);
        self.args[1] = try self.rig.input(self.dt, &.{@intCast(shape.k)}, self.signs);
        return true;
    }

    pub fn run(self: *Rq) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Rq) usize {
        return self.x.len;
    }

    pub fn reference(_: *const Rq, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The rule: v = H_256(signs * x) / 16 per block (fp32 butterflies: each value within gamma(8) of its block's
    /// |x| sum / 16); the row's scale is max |v| / 127 (NaN when the row holds one), each code v / scale rounded and
    /// clamped to +-127.
    pub fn check(self: *Rq, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const k = self.shape.k;
        const out = self.rig.outputBytes();
        const n = self.x.len;
        const v = try a.alloc(f64, k);
        defer a.free(v);
        const err_v = try a.alloc(f64, k / 256);
        defer a.free(err_v);
        var planted: usize = 0;
        var finite: usize = 0;
        for (0..self.shape.rows) |r| {
            var nan = false;
            var amax: f64 = 0;
            var emax: f64 = 0;
            for (0..k / 256) |b| {
                const blk = v[b * 256 ..][0..256];
                for (blk, 0..) |*o, j| o.* = self.x[r * k + b * 256 + j] * self.signs[b * 256 + j];
                err_v[b] = gamma(8) * host.sumAbs(blk) / 16;
                host.fwht(blk);
                for (blk) |*o| {
                    o.* /= 16;
                    if (std.math.isNan(o.*)) nan = true;
                    amax = @max(amax, @abs(o.*));
                }
                emax = @max(emax, err_v[b]);
            }
            const want = if (nan) std.math.nan(f64) else @max(amax / 127, 1e-20);
            const got = spec.read(.f32, out[n..], r);
            if (spec.classOf(want) != .finite) planted += 1 else finite += 1;
            if (spec.classOf(got) != spec.classOf(want)) return fmtMsg(&self.msg, "row {d}: scale {t} where float64 gives {t}", .{ r, spec.classOf(got), spec.classOf(want) });
            if (spec.classOf(want) != .finite) continue;
            if (@abs(got - want) > emax / 127 + 0x1p-22 * want) return fmtMsg(&self.msg, "row {d}: scale {e} where float64 gives {e}", .{ r, got, want });
            if (at != .shapes) continue;
            for (v, 0..) |x, i| {
                const code = host.intsOf(.int8, out, r * k + i);
                if (!codeOk(code, x / got, 0x1p-21, err_v[i / 256] / got)) return fmtMsg(&self.msg, "row {d} value {d}: code {d} where {e} / {e} rounds to {d}", .{ r, i, code, x, got, @round(x / got) });
            }
        }
        if (at == .nonfinite and (planted == 0 or finite == 0)) return "the planted values leave no row of one class: no proof";
        return null;
    }

    fn wrapper(self: *Rq) ![]const u8 {
        var act = try w8.quantize(self.args[0], self.rig.stream());
        defer act.deinit();
        return self.rig.bytesOf(&.{ act.xq, act.xs });
    }
};

fn rqSpec(comptime name: []const u8, known: ?spec.Known) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = zero,
        .out = .f32,
        .nan = .{ .stated = "a row holding a NaN has a NaN scale; codes exist for finite rows only" },
        // Eight 256-wide blocks a pass of the threadgroup.
        .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 256, .min = 256 } },
        .served = &.{ .{ .rows = 300, .n = 1, .k = 3072 }, .{ .rows = 64, .n = 1, .k = 12288 } },
        .known = known,
    };
}

// ── w8a8: the int8 GEMM ──

fn gemmLaunch(comptime name: [:0]const u8, comptime epi: [:0]const u8) metal.Launch {
    return .{ .name = name, .inputs = &.{ "xq", "wq", "xs", "ws" }, .outputs = &.{"y"}, .source = w8.GEMM_BODY ++ epi, .header = w8.HEADER, .tg_bytes = 0, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };
}
const gemm_plain = gemmLaunch("msv_w8a8_gemm", w8.EPI_PLAIN);
const gemm_swiglu = gemmLaunch("msv_w8a8_swiglu", w8.EPI_SWIGLU);

const Gemm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    swiglu: bool,
    dt: mlx.mlx_dtype = .bfloat16,
    shape: Shape = undefined,
    xq: []f64 = &.{},
    wq: []f64 = &.{},
    xs: []f64 = &.{},
    ws: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, swiglu: bool) Gemm {
        return .{ .rig = metal.Rig.init(a), .launch = if (swiglu) &gemm_swiglu else &gemm_plain, .swiglu = swiglu };
    }

    pub fn deinit(self: *Gemm) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.xq, self.wq, self.xs, self.ws }) |v| a.free(v);
    }

    fn outN(self: *const Gemm, shape: Shape) usize {
        return if (self.swiglu) shape.n / 2 else shape.n;
    }

    pub fn geometry(self: *const Gemm, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 64 != 0 or shape.k == 0 or shape.k % 256 != 0) return null;
        const m: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.n);
        const bm: c_int = if (m < 8192) 128 else 64;
        const bn: c_int = @divExact(8192, bm);
        var g = metal.Geometry{ .grid = .{ @divTrunc(n + bn - 1, bn) * 128, @divTrunc(m + bm - 1, bm), 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ m, @intCast(self.outN(shape)) }, self.dt);
        g.dtype("T", self.dt);
        g.int("BM", bm);
        g.int("BN", bn);
        return g;
    }

    pub fn prepare(self: *Gemm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        if (!w8.nax()) return error.NaxOnly;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.xq, &self.wq, &self.xs, &self.ws }, [_]usize{ shape.rows * shape.k, shape.n * shape.k, shape.rows, shape.n }) |p, len| {
            a.free(p.*);
            p.* = &.{};
            p.* = try a.alloc(f64, len);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        host.intsInto(rng, self.xq, -127, 127);
        host.intsInto(rng, self.wq, -127, 127);
        // Scales that put an output near N(0, 2): a SwiGLU gate stays within the bound's amplification (past -88 the
        // epilogue's exp(-g) overflows fp32 and silu flushes to -0, where the true value is a normal under 1e-36).
        const sk = 2 / (73 * 73 * @sqrt(@as(f64, @floatFromInt(shape.k))));
        host.uniformInto(rng, self.xs, 0.5 * sk, 1.5 * sk, .float32);
        host.uniformInto(rng, self.ws, 0.5, 1.5, .float32);
        if (fill == .nonfinite) {
            self.xs[0] = std.math.nan(f64);
            self.xs[1] = std.math.inf(f64);
            self.ws[2] = -std.math.inf(f64);
            self.ws[3] = std.math.nan(f64);
        }
        self.rig.reset();
        const m: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.n);
        const k: c_int = @intCast(shape.k);
        self.args[0] = try self.rig.input(.int8, &.{ m, k }, self.xq);
        self.args[1] = try self.rig.input(.int8, &.{ n, k }, self.wq);
        self.args[2] = try self.rig.input(.float32, &.{m}, self.xs);
        self.args[3] = try self.rig.input(.float32, &.{n}, self.ws);
        return true;
    }

    pub fn run(self: *Gemm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Gemm) usize {
        return self.shape.rows * self.outN(self.shape);
    }

    fn acc(self: *const Gemm, m: usize, n: usize) f64 {
        const k = self.shape.k;
        return host.dot(self.xq[m * k ..][0..k], self.wq[n * k ..][0..k])[0] * self.xs[m] * self.ws[n];
    }

    pub fn reference(self: *const Gemm, want: []f64, mag: []f64) !void {
        const on = self.outN(self.shape);
        for (0..self.shape.rows) |m| for (0..on) |j| {
            want[m * on + j] = if (self.swiglu) blk: {
                const g = self.acc(m, 2 * j);
                break :blk g / (1 + @exp(-g)) * self.acc(m, 2 * j + 1);
            } else self.acc(m, j);
            mag[m * on + j] = 0;
        };
    }

    fn wrapper(self: *Gemm) ![]const u8 {
        const act = w8.Act{ .xq = self.args[0], .xs = self.args[2] };
        const w = w8.Weight{ .wq = self.args[1], .ws = self.args[3], .n = @intCast(self.shape.n), .k = @intCast(self.shape.k) };
        const y = try w8.gemm(act, &w, self.dt, self.swiglu, self.rig.stream());
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

fn gemmSpec(comptime name: []const u8, swiglu: bool, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = zero,
        // int32 sums (exact) to float, times the row scale and the column scale; SwiGLU adds the fast exp (a few
        // ulps, times up to 1 + |g| <= 9 through silu), 1 +, the division and the product with up.
        .ops = if (swiglu) 64 else 3,
        .out = .bf16,
        .tiles = .{ .{ .size = 128 }, .{ .size = 64, .step = 64, .min = 64 }, .{} },
        .served = served,
    };
}

const scale_drops_nan: spec.Known = .{ .gate = .nonfinite, .finding = "the absmax runs through metal::max, which returns the other operand for a NaN: a block or row holding a NaN gets a finite scale (and an undefined code) where float64 gives NaN" };
const attn_short_keys: spec.Known = .{ .gate = .shapes, .finding = "the source binds `const device float* kbs = ks`; when kv heads x key blocks is under 8, MLX passes ks in constant space and the library fails to build (1 kv head at 32..224 keys)" };
const attn_drops_nan: spec.Known = .{ .gate = .nonfinite, .finding = "the channel max enters through metal::max(vm, 1e-20), which returns 1e-20 for a NaN: a NaN channel max gives finite outputs where float64 gives NaN" };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Qkv, .init(a, qwen_layout), qkvSpec("dit_qkv bf16 (Qwen-Image: 24 heads, f32 norm)"), null);
    try set.add(Qkv, .init(a, krea_layout), qkvSpec("dit_qkv bf16 (Krea: 24/8 heads, gate columns, rounded norm)"), null);
    const qkv_served: []const Shape = &.{ .{ .rows = 1024, .n = 4, .k = 128 }, .{ .rows = 333, .n = 2, .k = 128 } };
    try set.add(Quant, .init(a, &quant_contig, 1, 128, false, true, true, 1e-20), quantSpec("int8_attn_quant bf16 (q rows: 1 row, divide)", 1, qkv_served, scale_drops_nan), null);
    try set.add(Quant, .init(a, &quant_contig, 32, 128, true, true, true, 1e-20), quantSpec("int8_attn_quant bf16 (k blocks: 32 rows less the mean, divide)", 32, qkv_served, scale_drops_nan), null);
    const nax_served: []const Shape = &.{ .{ .rows = 200, .n = 4, .k = 256 }, .{ .rows = 333, .n = 2, .k = 256 } };
    try set.add(Quant, .init(a, &quant_strided, 64, 256, false, false, false, 1e-6), quantSpec("int8_attn_quant bf16 (strided q: 64 rows, D 256, multiply)", 64, nax_served, scale_drops_nan), null);
    try set.add(Quant, .init(a, &quant_strided, 32, 256, true, false, false, 1e-6), quantSpec("int8_attn_quant bf16 (strided k: 32 rows less the mean, D 256, multiply)", 32, nax_served, scale_drops_nan), null);
    try set.add(VPrep, .init(a), .{
        .name = "int8_attn_vprep bf16",
        .accum = .none,
        .depth = zero,
        .out = .f32,
        .nan = .{ .stated = "codes exist where v and its channel max are finite" },
        .served = &.{ .{ .rows = 1024, .n = 4, .k = 128 }, .{ .rows = 333, .n = 2, .k = 128 } },
    }, null);
    try set.add(Attn, .init(a, 8, 8, true), attnSpec("int8_attn bf16 (NAX; 8 heads)", attn_drops_nan), null);
    try set.add(Attn, .init(a, 2, 1, false), attnSpec("int8_attn bf16 (NAX; 2 heads over 1 kv head)", attn_short_keys), null);
    try set.add(Rq, .init(a), rqSpec("w8a8_rq bf16", scale_drops_nan), null);
    try set.add(Gemm, .init(a, false), gemmSpec("w8a8 gemm bf16 (NAX; BM 128)", false, &.{ .{ .rows = 129, .n = 256, .k = 3072 }, .{ .rows = 300, .n = 320, .k = 512 } }), null);
    try set.add(Gemm, .init(a, false), gemmSpec("w8a8 gemm bf16 (NAX; BM 64 past 8192 rows)", false, &.{.{ .rows = 8200, .n = 128, .k = 256 }}), null);
    try set.add(Gemm, .init(a, true), gemmSpec("w8a8 gemm + swiglu bf16 (NAX; BM 128)", true, &.{ .{ .rows = 129, .n = 256, .k = 3072 }, .{ .rows = 300, .n = 320, .k = 512 } }), null);
}

// ── tests ──

test "kernel contract dit: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const name = k.spec.name;
        if (std.mem.startsWith(u8, name, "int8_attn_vprep")) continue; // drifted through the attention's guard
        const nax_only = std.mem.indexOf(u8, name, "NAX") != null;
        if (nax_only and !w8.nax()) continue;
        const shape = k.spec.served[0];
        const ours, const theirs = if (std.mem.startsWith(u8, name, "dit_qkv"))
            try drift(Qkv, a, k.ctx, shape)
        else if (std.mem.startsWith(u8, name, "int8_attn_quant"))
            try drift(Quant, a, k.ctx, shape)
        else if (std.mem.startsWith(u8, name, "w8a8_rq"))
            try drift(Rq, a, k.ctx, shape)
        else if (std.mem.startsWith(u8, name, "w8a8 gemm"))
            try drift(Gemm, a, k.ctx, shape)
        else blk: {
            const t: *Attn = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try t.prepare(shape, .random, 7));
            break :blk try t.drift(a);
        };
        defer a.free(ours);
        defer if (std.mem.startsWith(u8, name, "int8_attn bf16")) a.free(theirs);
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}

fn drift(comptime T: type, a: std.mem.Allocator, ctx: *anyopaque, shape: Shape) !struct { []u8, []const u8 } {
    const t: *T = @ptrCast(@alignCast(ctx));
    try std.testing.expect(try t.prepare(shape, .random, 7));
    try t.run();
    const ours = try a.dupe(u8, t.rig.outputBytes());
    errdefer a.free(ours);
    return .{ ours, try t.wrapper() };
}
