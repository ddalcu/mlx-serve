//! Host values for registrations: inputs drawn and rounded to their dtype, the non-finite pattern gate 1 plants,
//! float64 sums and dots, a device array read back as float64, the quantized layouts kernels read (MLX's dense affine
//! packing, e2m1 nibbles with e4m3 or e8m0 group scales), and the forward-error allowance a reference carries when a
//! kernel rounds intermediates to bf16/f16 after fast-math arithmetic (`Val`, `allowance`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const spec = @import("spec.zig");

/// The output element a float dtype writes as (f32 for anything else).
pub fn elemOf(dt: mlx.mlx_dtype) spec.Elem {
    return switch (dt) {
        .float16 => .f16,
        .bfloat16 => .bf16,
        else => .f32,
    };
}

/// `v` as `dt` stores it: float dtypes round, integer ones keep the value.
pub fn rt(dt: mlx.mlx_dtype, v: f64) f64 {
    return switch (dt) {
        .float32, .float16, .bfloat16 => spec.roundTo(elemOf(dt), v),
        else => v,
    };
}

/// `n` values uniform in [lo, hi), rounded to `dt`.
pub fn uniform(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f64, hi: f64, dt: mlx.mlx_dtype) ![]f64 {
    const out = try a.alloc(f64, n);
    uniformInto(rng, out, lo, hi, dt);
    return out;
}

/// `uniform` into `out`.
pub fn uniformInto(rng: std.Random, out: []f64, lo: f64, hi: f64, dt: mlx.mlx_dtype) void {
    for (out) |*v| v.* = rt(dt, lo + (hi - lo) * rng.float(f64));
}

/// `n` integers uniform in [lo, hi] (exact in every float dtype for small ranges).
pub fn ints(a: std.mem.Allocator, rng: std.Random, n: usize, lo: i32, hi: i32) ![]f64 {
    const out = try a.alloc(f64, n);
    intsInto(rng, out, lo, hi);
    return out;
}

/// `ints` into `out`.
pub fn intsInto(rng: std.Random, out: []f64, lo: i32, hi: i32) void {
    for (out) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
}

/// Gate 1's pattern over `rows` rows of `width`: NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in
/// row 3 (rows past the buffer are skipped).
pub fn plant(x: []f64, rows: usize, width: usize) void {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, width - 1 } };
    const put = [_]f64{ nan, inf, -inf, inf, -inf };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * width + p[1]] = v;
    };
}

/// A float64 value and the most the device's value can differ from it (forward error, first order).
pub const Val = struct {
    v: f64,
    e: f64 = 0,

    pub fn exact(v: f64) Val {
        return .{ .v = v };
    }
};

/// f32 unit roundoff.
pub const U = 0x1p-24;

/// `x` rounded to `dt` where the device holds it within `x.e` of `x.v` before it rounds: the value float64 rounds to,
/// and the farthest any value in that interval rounds to from it (0 unless the interval straddles a rounding point).
pub fn round(dt: mlx.mlx_dtype, x: Val) Val {
    const r = rt(dt, x.v);
    if (x.e == 0 or !std.math.isFinite(x.v)) return .{ .v = r };
    return .{ .v = r, .e = @max(@abs(rt(dt, x.v + x.e) - r), @abs(rt(dt, x.v - x.e) - r)) };
}

/// Metal's fast `exp(x)` error relative to its value: 3 + floor(2|x|) f32 ulps (the MSL fast-math table).
pub fn fastExp(x: f64) f64 {
    return (3 + @floor(2 * @abs(x))) * 0x1p-23;
}

/// 1 / (1 + exp(-x)) as Metal computes it in f32 (fast exp, a fast divide), unrounded; `x` carries its own error.
pub fn sigmoid(x: Val) Val {
    const e = @exp(-x.v);
    const sig = 1 / (1 + e);
    if (!std.math.isFinite(x.v)) return .{ .v = sig };
    return .{ .v = sig, .e = sig * (1 - sig) * x.e + sig * (fastExp(x.v) * e * sig + U + 2.5 * 0x1p-23) };
}

/// x / (1 + exp(-x)) as Metal computes it in f32 (fast exp, a fast divide), unrounded; `x` carries its own error.
pub fn silu(x: Val) Val {
    const e = @exp(-x.v);
    const sig = 1 / (1 + e);
    const v = x.v * sig;
    if (!std.math.isFinite(x.v)) return .{ .v = v };
    return .{ .v = v, .e = @abs(sig * (1 + x.v * (1 - sig))) * x.e + @abs(v) * (fastExp(x.v) * e * sig + U + 2.5 * 0x1p-23) };
}

/// One ulp of `dt` relative to the value, at its widest.
pub fn ulpOf(dt: mlx.mlx_dtype) f64 {
    return switch (dt) {
        .bfloat16 => 0x1p-7,
        .float16 => 0x1p-10,
        else => 0x1p-23,
    };
}

/// `x` as the device may hold it: rounded to `dt` or not (an intermediate the compiler may keep in f32).
pub fn maybe(dt: mlx.mlx_dtype, x: Val) Val {
    const r = round(dt, x);
    return .{ .v = r.v, .e = @max(r.e, x.e + @abs(x.v - r.v)) };
}

/// a * b before any rounding.
pub fn mul(a: Val, b: Val) Val {
    return .{ .v = a.v * b.v, .e = @abs(a.v) * b.e + @abs(b.v) * a.e + a.e * b.e };
}

/// MLX's unary sigmoid in `dt`: sy = 1 / (1 + exp(|x|)), x < 0 ? sy : 1 - sy, each step in `dt` or in f32, with a
/// fast-math or a precise (4 ulps) exp.
pub fn mlxSig(dt: mlx.mlx_dtype, x: Val) Val {
    const ax = @abs(x.v);
    if (!std.math.isFinite(x.v)) {
        const sy = rt(dt, 1 / (1 + @exp(ax)));
        return .{ .v = if (x.v < 0) sy else rt(dt, 1 - sy) };
    }
    const ev = @exp(ax);
    const e = maybe(dt, .{ .v = ev, .e = ev * (std.math.expm1(x.e) + @max(fastExp(ax), 4 * 0x1p-23, ulpOf(dt))) });
    const d = maybe(dt, .{ .v = 1 + e.v, .e = e.e + U * (1 + e.v) });
    const r = 1 / d.v;
    var sy = round(dt, .{ .v = r, .e = d.e / (d.v * @max(d.v - d.e, 1)) + r * @max(2.5 * 0x1p-23, ulpOf(dt)) });
    // Within its error of 0 the device may take the other branch: the two meet at 1/2.
    const straddle = if (ax <= x.e) @abs(1 - 2 * sy.v) + sy.e else 0;
    if (x.v < 0) {
        sy.e += straddle;
        return sy;
    }
    var om = round(dt, maybe(dt, .{ .v = 1 - sy.v, .e = sy.e + U * @abs(1 - sy.v) }));
    om.e += straddle;
    return om;
}

/// The mag that makes gate 3's bound equal `allow` for a spec of depth 1 with f32 sums, no ops and no mids: a
/// reference that carries its own forward error (`Val`) registers that way.
pub fn allowance(allow: f64) f64 {
    return allow * (1 - U) / U;
}

/// Depth 1, for registrations whose reference carries the whole allowance (`allowance`).
pub fn depthOne(_: spec.Shape) usize {
    return 1;
}

/// Terms summed in float64 in order (the reference's sums).
pub fn sum(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

/// The terms' magnitude: the bound's `mag`.
pub fn sumAbs(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += @abs(v);
    return s;
}

/// The sum of x[i] * w[i] and of its terms' magnitudes, in float64 (vectorized: references run in Debug builds).
pub fn dot(x: []const f64, w: []const f64) [2]f64 {
    const V = @Vector(8, f64);
    var s: V = @splat(0);
    var m: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const p = @as(V, x[i..][0..8].*) * @as(V, w[i..][0..8].*);
        s += p;
        m += @abs(p);
    }
    var out = [2]f64{ @reduce(.Add, s), @reduce(.Add, m) };
    while (i < x.len) : (i += 1) {
        out[0] += x[i] * w[i];
        out[1] += @abs(x[i] * w[i]);
    }
    return out;
}

/// The Walsh-Hadamard transform of `v` in place (Sylvester order, unscaled), in float64.
pub fn fwht(v: []f64) void {
    var h: usize = 1;
    while (h < v.len) : (h <<= 1) {
        var i: usize = 0;
        while (i < v.len) : (i += 2 * h) for (i..i + h) |j| {
            const a = v[j];
            const b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        };
    }
}

/// An affine-quantized [rows, cols] weight as MLX packs it: `bits`-bit codes little-endian in u32 words (a dense
/// bitstream per row, `mx.quantize`'s layout at every width), scales and biases per `gs` columns in `dt`, and the
/// dequantized values (code * scale + bias, float64). Bits 16: the words hold the raw `dt` values, two a word.
pub const Quant = struct { words: []f64, scales: []f64, biases: []f64, w: []f64 };

pub fn quantize(a: std.mem.Allocator, rng: std.Random, rows: usize, cols: usize, bits: u32, gs: usize, dt: mlx.mlx_dtype) !Quant {
    const per_row = cols * bits / 32;
    var q: Quant = .{
        .words = try a.alloc(f64, rows * per_row),
        .scales = try a.alloc(f64, if (bits == 16) 1 else rows * (cols / gs)),
        .biases = try a.alloc(f64, if (bits == 16) 1 else rows * (cols / gs)),
        .w = try a.alloc(f64, rows * cols),
    };
    @memset(q.words, 0);
    @memset(q.scales, 1);
    @memset(q.biases, 0);
    const top: u32 = if (bits == 16) 0xffff else (@as(u32, 1) << @intCast(bits)) - 1;
    if (bits != 16) for (q.scales, q.biases) |*sc, *bi| {
        sc.* = rt(dt, (0.5 + rng.float(f64)) * 0.02);
        bi.* = rt(dt, -sc.* * @as(f64, @floatFromInt(top)) / 2 * (0.8 + 0.4 * rng.float(f64)));
    };
    for (0..rows) |r| for (0..cols) |c| {
        var code: u32 = rng.uintAtMost(u32, top);
        if (bits == 16) {
            const v = rt(dt, (rng.float(f64) * 2 - 1) * 0.1);
            code = if (dt == .bfloat16) spec.toBf16(@floatCast(v)) else @as(u16, @bitCast(@as(f16, @floatCast(v))));
            q.w[r * cols + c] = v;
        } else {
            const g = r * (cols / gs) + c / gs;
            q.w[r * cols + c] = @as(f64, @floatFromInt(code)) * q.scales[g] + q.biases[g];
        }
        setCode(q.words[r * per_row ..][0..per_row], c, bits, code);
    };
    return q;
}

/// Writes `code` as element `i` of a little-endian `bits`-bit stream held in u32 words (as float64 values).
pub fn setCode(words: []f64, i: usize, bits: u32, code: u32) void {
    var bit = i * bits;
    var left = bits;
    var v = code;
    while (left > 0) {
        const w = bit / 32;
        const off: u5 = @intCast(bit % 32);
        const take = @min(left, 32 - @as(u32, off));
        const mask: u64 = (@as(u64, 1) << @intCast(take)) - 1;
        var word: u64 = @intFromFloat(words[w]);
        word |= (@as(u64, v) & mask) << off;
        words[w] = @floatFromInt(word);
        v = if (take < 32) v >> @intCast(take) else 0;
        bit += take;
        left -= take;
    }
}

/// Re-derives the dequantized values after codes, scales or biases were edited in place (a stress fill).
pub fn dequant(q: *Quant, rows: usize, cols: usize, bits: u32, gs: usize, codes: []const u32) void {
    const per_row = cols * bits / 32;
    @memset(q.words, 0);
    for (0..rows) |r| for (0..cols) |c| {
        const code = codes[r * cols + c];
        const g = r * (cols / gs) + c / gs;
        q.w[r * cols + c] = @as(f64, @floatFromInt(code)) * q.scales[g] + q.biases[g];
        setCode(q.words[r * per_row ..][0..per_row], c, bits, code);
    };
}

/// `codes` (each below 2^bits) packed as MLX's `quantize` packs a row (`setCode`), a row `k * bits / 32` words.
pub fn packAffine(a: std.mem.Allocator, codes: []const u8, rows: usize, k: usize, bits: u32) ![]f64 {
    const wpr = k * bits / 32;
    const out = try a.alloc(f64, rows * wpr);
    @memset(out, 0);
    for (0..rows) |r| for (0..k) |i| setCode(out[r * wpr ..][0..wpr], i, bits, codes[r * k + i]);
    return out;
}

/// An affine-quantized [n, k] weight as MLX's `quantize` lays it out: codes, a scale and bias per `gs` group (in the
/// activation dtype), and the packed words the kernels read.
pub const Affine = struct {
    bits: u32,
    gs: usize,
    n: usize,
    k: usize,
    codes: []u8 = &.{},
    scales: []f64 = &.{},
    biases: []f64 = &.{},
    words: []f64 = &.{},

    pub fn deinit(self: *Affine, a: std.mem.Allocator) void {
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.biases);
        a.free(self.words);
        self.* = .{ .bits = self.bits, .gs = self.gs, .n = 0, .k = 0 };
    }

    /// Codes, scales and biases for `n` x `k`: random (a centered affine grid with scales near 0.01), or the stress
    /// form (integer codes mirrored across each row's halves, scale 1, bias 0), then packed.
    pub fn fill(self: *Affine, a: std.mem.Allocator, rng: std.Random, n: usize, k: usize, dt: mlx.mlx_dtype, stress: bool) !void {
        self.deinit(a);
        self.n = n;
        self.k = k;
        self.codes = try a.alloc(u8, n * k);
        self.scales = try a.alloc(f64, n * k / self.gs);
        self.biases = try a.alloc(f64, n * k / self.gs);
        const top: u32 = @as(u32, 1) << @intCast(self.bits);
        rng.bytes(self.codes);
        if (stress) {
            // Codes under 16 at every width, so a row's |x q| stays under 2^24 at K 6144 (fp32 then sums exactly).
            const h = k / 2;
            for (0..n) |r| for (0..h) |j| {
                const c: u8 = self.codes[r * k + j] & 15 & @as(u8, @intCast(top - 1));
                self.codes[r * k + j] = c;
                self.codes[r * k + h + j] = c;
            };
            @memset(self.scales, 1);
            @memset(self.biases, 0);
        } else {
            for (self.codes) |*c| c.* &= @intCast(top - 1);
            const half: f64 = @floatFromInt(top / 2);
            for (self.scales, self.biases) |*s, *b| {
                s.* = rt(dt, (0.005 + 0.01 * rng.float(f64)) * 16 / half);
                b.* = rt(dt, -s.* * half * (0.8 + 0.4 * rng.float(f64)));
            }
        }
        self.words = try packAffine(a, self.codes, n, k, self.bits);
    }

    /// Row `r` dequantized (q * s + b) into `w`, and each value's magnitude through the dequant (|q s| + |b|).
    pub fn row(self: *const Affine, r: usize, w: []f64, m: []f64) void {
        for (w, m, 0..) |*o, *g, i| {
            const at = r * self.k + i;
            const s = self.scales[at / self.gs];
            const b = self.biases[at / self.gs];
            const q: f64 = @floatFromInt(self.codes[at]);
            o.* = q * s + b;
            g.* = @abs(q * s) + @abs(b);
        }
    }
};

/// `codes` (4-bit) packed eight to a uint32, value i at bit 4 * i.
pub fn packNibbles(a: std.mem.Allocator, codes: []const u8) ![]f64 {
    const out = try a.alloc(f64, codes.len / 8);
    for (out, 0..) |*w, j| {
        var v: u32 = 0;
        for (0..8) |i| v |= @as(u32, codes[8 * j + i] & 0xf) << @intCast(4 * i);
        w.* = @floatFromInt(v);
    }
    return out;
}

/// An e2m1 code's value (sign bit 3, two exponent bits, one mantissa bit; no infinities or NaN).
pub fn e2m1(code: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = mag[code & 7];
    return if (code & 8 != 0) -v else v;
}

/// An e4m3 code's value (bias 7, subnormals at exponent 0); 0x7f / 0xff are NaN in the format, never drawn here.
pub fn e4m3(code: u8) f64 {
    return e4m3_table[code];
}

const e4m3_table = blk: {
    @setEvalBranchQuota(100_000);
    var t: [256]f64 = undefined;
    for (&t, 0..) |*o, c| {
        const e: i32 = (c >> 3) & 0xf;
        const m: f64 = @floatFromInt(c & 7);
        const v = if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
        o.* = if (c & 0x80 != 0) -v else v;
    }
    break :blk t;
};

/// An e8m0 scale's value: 2^(code - 127).
pub fn e8m0(code: u8) f64 {
    return std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, code) - 127)));
}

/// A random e4m3 code with a value in [2^lo, 2^hi), either sign when `signed`.
pub fn e4m3Code(rng: std.Random, lo: i32, hi: i32, signed: bool) u8 {
    const e: u8 = @intCast(lo + 7 + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo)))));
    const code: u8 = (e << 3) | rng.uintLessThan(u8, 8);
    return if (signed and rng.boolean()) code | 0x80 else code;
}

/// Integers whose halves of each `n`-long row mirror: the halves sum past fp16's range, the row to a small integer (the
/// +1 offsets sit at 32..39, so a value made large later stays exactly mirrored).
pub fn mirrored(rng: std.Random, v: []f64, n: usize) void {
    const h = n / 2;
    for (0..v.len / n) |r| for (0..h) |j| {
        const x: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
        v[r * n + j] = x;
        v[r * n + h + j] = -x + @as(f64, if (j >= 32 and j < 40) 1 else 0);
    };
}

/// `mirrored` for a butterfly, whose sums also run in 32 strided lanes: each block's first 32 values are multiples of
/// 16 past 2048 (one a lane, exact in bf16), so every lane's partial sum lands off fp16's grid too.
pub fn mirroredWide(rng: std.Random, v: []f64, n: usize) void {
    mirrored(rng, v, n);
    const h = n / 2;
    for (0..v.len / n) |r| for (0..32) |j| {
        const big: f64 = @floatFromInt(16 * (128 + rng.uintLessThan(u32, 128)));
        v[r * n + j] = big;
        v[r * n + h + j] = -big;
    };
}

/// Gates 1 and 3 for a kernel with per-output allowances: at `.nonfinite` every output's class equals float64's
/// (both kinds present), at `.shapes` also |got - want| <= allow. Null when it holds, else what failed (in `buf`).
pub fn within(buf: []u8, at: spec.Gate, got: []const f64, want: []const f64, allow: []const f64) ?[]const u8 {
    var planted: usize = 0;
    for (got, want, allow, 0..) |g, w, l, i| {
        if (spec.classOf(w) != .finite) planted += 1;
        if (spec.classOf(g) != spec.classOf(w))
            return std.fmt.bufPrint(buf, "output {d} is {t} where float64 gives {t}", .{ i, spec.classOf(g), spec.classOf(w) }) catch buf;
        if (at == .shapes and spec.classOf(w) == .finite and !(@abs(g - w) <= l))
            return std.fmt.bufPrint(buf, "output {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its allowance", .{ i, g, w, @abs(g - w) / l }) catch buf;
    }
    if (at == .nonfinite and (planted == 0 or planted == want.len))
        return std.fmt.bufPrint(buf, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ planted, want.len }) catch buf;
    return null;
}

/// A 0-dim int32 input (a row count): MLX passes it by value; the rig frees it with the other inputs.
pub fn intScalar(rig: anytype, v: i32) !mlx.mlx_array {
    const x = mlx.mlx_array_new_int(v);
    try rig.ins.append(rig.a, .{ .parent = x, .view = mlx.mlx_array_new(), .len = 1 });
    return x;
}

/// An evaluated copy of `x`'s values (float, int8 or 32-bit integer dtypes) as float64.
pub fn read(a: std.mem.Allocator, x: mlx.mlx_array, s: mlx.mlx_stream) ![]f64 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const raw = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    const out = try a.alloc(f64, n);
    errdefer a.free(out);
    const dt = mlx.mlx_array_dtype(c);
    for (out, 0..) |*v, i| v.* = switch (dt) {
        .float32 => spec.read(.f32, raw[0 .. 4 * n], i),
        .float16 => spec.read(.f16, raw[0 .. 2 * n], i),
        .bfloat16 => spec.read(.bf16, raw[0 .. 2 * n], i),
        .int8, .uint8, .int32, .uint32 => intsOf(dt, raw[0 .. n * mlx.mlx_array_itemsize(c)], i),
        else => return error.UnsupportedDtype,
    };
    return out;
}

/// Element `i` of little-endian `bytes` holding an integer dtype, as f64 (an int8 / uint32 output's values).
pub fn intsOf(dt: mlx.mlx_dtype, bytes: []const u8, i: usize) f64 {
    return switch (dt) {
        .int8 => @floatFromInt(@as(i8, @bitCast(bytes[i]))),
        .uint8 => @floatFromInt(bytes[i]),
        .int32 => @floatFromInt(std.mem.readInt(i32, bytes[4 * i ..][0..4], .little)),
        .uint32 => @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        else => unreachable,
    };
}

// ── tests ──

test "kernel contract host: values round to their dtype and the planted pattern lands where it says" {
    var prng = std.Random.DefaultPrng.init(1);
    const v = try uniform(std.testing.allocator, prng.random(), 64, -3, 3, .bfloat16);
    defer std.testing.allocator.free(v);
    for (v) |x| try std.testing.expectEqual(spec.roundTo(.bf16, x), x);
    var x: [16]f64 = @splat(0);
    plant(&x, 4, 4);
    try std.testing.expect(std.math.isNan(x[1]) and x[6] == std.math.inf(f64) and x[11] == -std.math.inf(f64));
    try std.testing.expect(x[12] == std.math.inf(f64) and x[15] == -std.math.inf(f64));
}

test "kernel contract host: codes pack as a little-endian bitstream at every width" {
    for ([_]u32{ 3, 4, 5, 6, 8 }) |bits| {
        var words: [15]f64 = @splat(0);
        const n = words.len * 32 / bits;
        for (0..n) |i| setCode(&words, i, bits, @intCast((i * 7 + 3) % (@as(usize, 1) << @intCast(bits))));
        for (0..n) |i| {
            var got: u64 = 0;
            for (0..bits) |b| {
                const at = i * bits + b;
                const word: u32 = @intFromFloat(words[at / 32]);
                got |= @as(u64, (word >> @intCast(at % 32)) & 1) << @intCast(b);
            }
            try std.testing.expectEqual(@as(u64, (i * 7 + 3) % (@as(usize, 1) << @intCast(bits))), got);
        }
    }
}

test "kernel contract host: dense affine packing straddles words at 3, 5 and 6 bits like mx.quantize" {
    const a = std.testing.allocator;
    var codes: [64]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i % 8);
    const w = try packAffine(a, &codes, 1, 64, 3);
    defer a.free(w);
    try std.testing.expectEqual(@as(usize, 6), w.len);
    // value 10 (code 2) sits at bit 30: one bit in word 0, its high bit in word 1.
    const w0: u32 = @intFromFloat(w[0]);
    const w1: u32 = @intFromFloat(w[1]);
    try std.testing.expectEqual(@as(u32, 2), (w0 >> 30) | ((w1 & 1) << 2));
    try std.testing.expectEqual(@as(f64, 448), e4m3(0x7e));
    try std.testing.expectEqual(@as(f64, -6), e2m1(0xf));
    try std.testing.expectEqual(@as(f64, 0x1p-9), e4m3(1));
}
