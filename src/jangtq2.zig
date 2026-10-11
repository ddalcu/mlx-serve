// SPDX-License-Identifier: Apache-2.0
// Ported from vMLX (jjang-ai/vmlx) vmlx_engine/jangh/kernels.py and switch.py @ 5f007e6d.
//! JANGTQ v2 routed experts (`jangtq2`, the `switch_mlp` banks of JANGH bundles). A weight row is
//! `scale[row] * level(q)`: codes in MLX's affine bit packing, `level` an odd cubic of the code
//! with its constants baked into the Metal (`CODEBOOK`). A rotated bank was quantized against a
//! blockwise normalized Hadamard-32 of its input, which the activations get before each projection.
//! `moe` is vMLX's TQSwitchGLU.routed() with the settings its qwen4_exp loader uses, bit for bit.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const mlx_headers = @import("mlx_steel_sources");

const Allocator = std.mem.Allocator;

/// One projection's experts: `packed_w` uint32 [E, N, K*bits/32] (LSB bitstream), `scales` float16 [E, N].
pub const Proj = struct { packed_w: mlx.mlx_array, scales: mlx.mlx_array };
/// `rotated`: every projection was quantized with rotation "hadamard32" (false: "none").
pub const Bank = struct { gate: Proj, up: Proj, down: Proj, rotated: bool };

/// Up to this many tokens take the gather (qmv) kernels, more the expert-sorted GEMM. A count of
/// tokens, not routed rows (runtime_identity.QWEN4_DECODE_MAX_TOKENS).
pub const DECODE_MAX_TOKENS = 96;

const NSG = 2; // decode: simdgroups per threadgroup
const RPS = 4; // decode: rows per simdgroup
const VPT = 16; // decode: values per lane per block
const BLOCK = VPT * 32; // decode: values per block per simdgroup
const UNROLL = "_Pragma(\"clang loop unroll(full)\")";

/// `tq_level<bits>(q)`, the v2 odd-cubic codebook as literal immediates (kernels.py _cb_header): the
/// spellings are Python's repr() of contract.CUBIC_PARAMS, so the compiler folds the same constants.
const CODEBOOK =
    \\template <int bits> METAL_FUNC float tq_level(uint q);
    \\template <> METAL_FUNC float tq_level<2>(uint q) { float lo = (q & 1u) ? -0.4528312385082245f : -1.5104438066482544f; float hi = (q & 1u) ? 1.5104438066482544f : 0.4528312385082245f; return (q & 2u) ? hi : lo; }
    \\template <> METAL_FUNC float tq_level<3>(uint q) { float u = float(q) - 3.5f; return u * fma(0.011599999999999997f, u * u, 0.47124999999999995f); }
    \\template <> METAL_FUNC float tq_level<4>(uint q) { float u = float(q) - 7.5f; return u * fma(0.002100000000000002f, u * u, 0.2405f); }
    \\template <> METAL_FUNC float tq_level<6>(uint q) { float u = float(q) - 31.5f; return u * fma(0.0f, u * u, 0.10413533834586466f); }
    \\template <> METAL_FUNC float tq_level<8>(uint q) { float u = float(q) - 127.5f; return u * fma(0.0f, u * u, 0.030780075187969925f); }
    \\
;

/// The sorted prefill GEMM: NAX tiles on M5-class GPUs, MLX's steel tiles elsewhere.
const Prefill = enum { nax, steel };
var decode_logged = false;
var prefill_logged = false;

const Geometry = struct { tokens: c_int, k: c_int, d: c_int, i: c_int, e: c_int, bits_gu: u32, bits_dn: u32 };

/// TQSwitchGLU.routed(x, inds, scores): the router-weighted sum of the routed experts, no shared
/// expert. x [..., D] bf16/f16/f32, inds [..., k] any integer dtype (each < E), scores [..., k]
/// f16/bf16/f32; returns [..., D] in x's dtype. `limit` > 0 clamps SwiGLU (gate above, up both ways).
/// The k experts of a token are summed in the order `inds` lists them. f32 x (QWEN4_STREAM_F32)
/// prefills on the steel tiles even where NAX is available: the fused f32 NAX tile does not fit.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, limit: f32) !mlx.mlx_array {
    const nax = @import("transformer.zig").naxAvailable() and mlx.mlx_array_dtype(x) != .float32;
    return moeOn(s, x, bank, inds, scores, limit, if (nax) .nax else .steel);
}

/// `moe` with the prefill GEMM on `arm`.
fn moeOn(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, limit: f32, arm: Prefill) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.Jangtq2NeedsMetal;
    const g = try geometry(x, bank, inds, scores);
    const rows = g.tokens * g.k;
    const x2 = try reshape(x, &.{ g.tokens, g.d }, s);
    defer free(x2);
    const flat = try reshape(inds, &.{rows}, s);
    defer free(flat);
    const idx = try astype(flat, .uint32, s);
    defer free(idx);
    const sflat = try reshape(scores, &.{rows}, s);
    defer free(sflat);
    const wts = try astype(sflat, .float32, s);
    defer free(wts);
    const y = if (g.tokens > DECODE_MAX_TOKENS)
        try prefill(s, x2, bank, idx, wts, g, limit, arm)
    else
        try decode(s, x2, bank, idx, wts, g, limit);
    defer free(y);
    return reshape(y, mlx.getShape(x), s);
}

fn geometry(x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array) !Geometry {
    if (typeName(mlx.mlx_array_dtype(x)) == null) return error.Jangtq2Dtype;
    switch (mlx.mlx_array_dtype(inds)) {
        .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64 => {},
        else => return error.Jangtq2Dtype,
    }
    switch (mlx.mlx_array_dtype(scores)) {
        .float16, .bfloat16, .float32 => {},
        else => return error.Jangtq2Dtype,
    }
    const xs = mlx.getShape(x);
    const is = mlx.getShape(inds);
    if (xs.len == 0 or is.len != xs.len or !std.mem.eql(c_int, is, mlx.getShape(scores))) return error.Jangtq2Shape;
    if (!std.mem.eql(c_int, xs[0 .. xs.len - 1], is[0 .. is.len - 1])) return error.Jangtq2Shape;
    var tokens: c_int = 1;
    for (xs[0 .. xs.len - 1]) |v| tokens = std.math.mul(c_int, tokens, v) catch return error.Jangtq2Shape;
    const d = xs[xs.len - 1];
    const k = is[is.len - 1];
    if (tokens <= 0 or d <= 0 or k <= 0) return error.Jangtq2Shape;
    _ = std.math.mul(c_int, tokens, k) catch return error.Jangtq2Shape;
    const gs = mlx.getShape(bank.gate.packed_w);
    if (gs.len != 3 or gs[0] <= 0 or gs[1] <= 0) return error.Jangtq2Shape;
    const e = gs[0];
    const i = gs[1];
    if (@rem(d, 32) != 0 or @rem(i, 32) != 0) return error.Jangtq2Shape;
    try checkProj(bank.gate, e, i);
    try checkProj(bank.up, e, i);
    try checkProj(bank.down, e, d);
    if (!std.mem.eql(c_int, gs, mlx.getShape(bank.up.packed_w))) return error.Jangtq2Shape;
    return .{
        .tokens = tokens,
        .k = k,
        .d = d,
        .i = i,
        .e = e,
        .bits_gu = try bitsOf(gs[2], d),
        .bits_dn = try bitsOf(mlx.getShape(bank.down.packed_w)[2], i),
    };
}

fn checkProj(p: Proj, e: c_int, n: c_int) !void {
    if (mlx.mlx_array_dtype(p.packed_w) != .uint32 or mlx.mlx_array_dtype(p.scales) != .float16) return error.Jangtq2Dtype;
    const ps = mlx.getShape(p.packed_w);
    if (ps.len != 3 or ps[0] != e or ps[1] != n or ps[2] <= 0) return error.Jangtq2Shape;
    if (!std.mem.eql(c_int, mlx.getShape(p.scales), &.{ e, n })) return error.Jangtq2Shape;
}

/// The code width a packed row of `words` uint32 implies for `cols` inputs.
fn bitsOf(words: c_int, cols: c_int) !u32 {
    const total = @as(i64, words) * 32;
    if (@rem(total, cols) != 0) return error.Jangtq2Bits;
    return switch (@divExact(total, cols)) {
        inline 2, 3, 4, 6, 8 => |b| b,
        else => error.Jangtq2Bits,
    };
}

/// routed()'s gather branch: rotate x, fused gate/up/SwiGLU (f32 rows), rotate, down with the
/// router-weighted sum fused.
fn decode(s: mlx.mlx_stream, x2: mlx.mlx_array, bank: Bank, idx: mlx.mlx_array, wts: mlx.mlx_array, g: Geometry, limit: f32) !mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x2);
    if (!decode_logged) {
        decode_logged = true;
        log.info("[jangtq2] decode engaged: gather qmv E={d} D={d} I={d} bits={d}/{d} rotated={any} (<= {d} tokens)\n", .{ g.e, g.d, g.i, g.bits_gu, g.bits_dn, bank.rotated, DECODE_MAX_TOKENS });
    }
    const xr = if (bank.rotated) try h32Rows(s, x2, xd) else try dup(x2);
    defer free(xr);
    const h = try gatherQmv(s, xr, bank.gate, bank.up, idx, g.bits_gu, limit);
    defer free(h);
    const hr = if (bank.rotated) try h32Rows(s, h, .float32) else try dup(h);
    defer free(hr);
    return gatherQmvWeightedDown(s, hr, bank.down, idx, wts, g.tokens, g.k, g.bits_dn, xd);
}

/// switch.py _prefill with use_weighted_unsort: rows sorted by expert, fused gate/up GEMM, rotate,
/// down GEMM, then the fp32 weighted sum back in token order.
fn prefill(s: mlx.mlx_stream, x2: mlx.mlx_array, bank: Bank, idx: mlx.mlx_array, wts: mlx.mlx_array, g: Geometry, limit: f32, arm: Prefill) !mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x2);
    // The fused f32 NAX tile needs 34816 bytes of threadgroup memory (32768 max).
    if (arm == .nax and xd == .float32) return error.Jangtq2Dtype;
    if (@rem(g.d, 64) != 0 or @rem(g.i, 64) != 0) return error.Jangtq2Shape;
    if (!prefill_logged) {
        prefill_logged = true;
        log.info("[jangtq2] prefill engaged: {s} sorted GEMM E={d} D={d} I={d} bits={d}/{d} rotated={any} (> {d} tokens)\n", .{ @tagName(arm), g.e, g.d, g.i, g.bits_gu, g.bits_dn, bank.rotated, DECODE_MAX_TOKENS });
    }
    var r = try @import("transformer.zig").sortRoutes(s, idx, g.k);
    defer r.deinit();
    const xr = if (bank.rotated) try h32Rows(s, x2, xd) else try dup(x2);
    defer free(xr);
    var xs = mlx.mlx_array_new();
    defer free(xs);
    try mlx.check(mlx.mlx_take_axis(&xs, xr, r.lhs, 0, s));
    const h = try gatherQmmSorted(s, xs, bank.gate, bank.up, r.sorted, g.bits_gu, limit, arm);
    defer free(h);
    const hr = if (bank.rotated) try h32Rows(s, h, xd) else try dup(h);
    defer free(hr);
    const y = try gatherQmmSorted(s, hr, bank.down, null, r.sorted, g.bits_dn, 0, arm);
    defer free(y);
    return weightedUnsort(s, y, r.inverse, wts, g.tokens, g.k);
}

/// kernels.py h32_rows: blockwise normalized Hadamard-32 over the last axis of x [M, K], one launch,
/// result in `out`.
fn h32Rows(s: mlx.mlx_stream, x: mlx.mlx_array, out: mlx.mlx_dtype) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const m = sh[0];
    const k = sh[1];
    if (@rem(k, 32) != 0) return error.Jangtq2Shape;
    const kern = try kernelFor(.{ .kind = .h32, .a = mlx.mlx_array_dtype(x), .b = out });
    const meta = try consts(.uint32, &.{ @intCast(k), 0, 0, 0, 0, 0, 0, 0 });
    return run(kern, &.{ x, meta }, .{ .grid = .{ 32, @divExact(k, 32), m }, .group = .{ 32, 1, 1 }, .shape = &.{ m, k }, .dtype = out }, s);
}

/// kernels.py _tail_mode: "al" unguarded, "tl" unguarded main blocks + one guarded tail block
/// (K % BLOCK != 0). Rows always fill whole simdgroups: `geometry` admits N % 32 == 0 only.
const Mode = enum { al, tl };
fn tailMode(k: c_int) Mode {
    return if (@rem(k, BLOCK) == 0) .al else .tl;
}

/// kernels.py gather_qmv with gate and up fused: x [nx, K], idx [ndisp] uint32 (the expert per
/// dispatch; dispatch d reads x row d / (ndisp / nx)) -> f32 [ndisp, N] = SwiGLU(Wg_e x, Wu_e x),
/// clamped at `limit` > 0.
fn gatherQmv(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, up: Proj, idx: mlx.mlx_array, bits: u32, limit: f32) !mlx.mlx_array {
    const nx = mlx.getShape(x)[0];
    const k = mlx.getShape(x)[1];
    const n = mlx.getShape(gate.packed_w)[1];
    const ndisp: c_int = @intCast(mlx.mlx_array_size(idx));
    if (@rem(k, 32) != 0) return error.Jangtq2Shape;
    const kern = try kernelFor(.{ .kind = .qmv, .bits = bits, .a = mlx.mlx_array_dtype(x), .mode = tailMode(k) });
    const meta = try consts(.uint32, &.{ @intCast(k), @intCast(n), @intCast(@divTrunc(ndisp, nx)) });
    const lim = try consts(.float32, &.{@bitCast(limit)});
    return run(kern, &.{ x, gate.packed_w, gate.scales, up.packed_w, up.scales, idx, meta, lim }, .{
        .grid = .{ NSG * 32, @divTrunc(n + NSG * RPS - 1, NSG * RPS), ndisp },
        .group = .{ NSG * 32, 1, 1 },
        .shape = &.{ ndisp, n },
        .dtype = .float32,
    }, s);
}

/// kernels.py gather_qmv_weighted_down: h [T*k, K] per dispatch, idx/wts [T*k] (wts f32) ->
/// [T, N] = sum over a token's k dispatches of wts * (W_e h), in `out`.
fn gatherQmvWeightedDown(s: mlx.mlx_stream, h: mlx.mlx_array, d: Proj, idx: mlx.mlx_array, wts: mlx.mlx_array, tokens: c_int, k: c_int, bits: u32, out: mlx.mlx_dtype) !mlx.mlx_array {
    const kin = mlx.getShape(h)[1];
    const n = mlx.getShape(d.packed_w)[1];
    if (@rem(kin, 32) != 0) return error.Jangtq2Shape;
    const kern = try kernelFor(.{ .kind = .wdown, .bits = bits, .a = out, .b = mlx.mlx_array_dtype(h), .mode = tailMode(kin) });
    const meta = try consts(.uint32, &.{ @intCast(kin), @intCast(n), @intCast(k) });
    return run(kern, &.{ h, d.packed_w, d.scales, idx, wts, meta }, .{
        .grid = .{ NSG * 32, @divTrunc(n + NSG * RPS - 1, NSG * RPS), tokens },
        .group = .{ NSG * 32, 1, 1 },
        .shape = &.{ tokens, n },
        .dtype = out,
    }, s);
}

/// kernels.py gather_qmm_sorted: x [M, K] rows sorted by expert (`idx` [M] uint32) -> [M, N] in x's
/// dtype. Single: x W_e^T; with `up`: SwiGLU(x Wg_e^T, x Wu_e^T) clamped at `limit` > 0. M is past 96
/// tokens' routes, so `idx` is never under 8 entries (MLX binds those in `constant` space).
fn gatherQmmSorted(s: mlx.mlx_stream, x: mlx.mlx_array, w: Proj, up: ?Proj, idx: mlx.mlx_array, bits: u32, limit: f32, arm: Prefill) !mlx.mlx_array {
    const m = mlx.getShape(x)[0];
    const k = mlx.getShape(x)[1];
    const n = mlx.getShape(w.packed_w)[1];
    if (@rem(k, 64) != 0) return error.Jangtq2Shape;
    const xd = mlx.mlx_array_dtype(x);
    const meta = mlx.mlx_array_new_data(&[_]i32{ m, n, k }, &[_]c_int{3}, 1, .int32);
    defer free(meta);
    if (arm == .steel) {
        const gate = try steelSorted(s, x, w, idx, meta, bits);
        const u = up orelse return gate;
        defer free(gate);
        return steelSwiglu(s, gate, try steelSorted(s, x, u, idx, meta, bits), limit, xd);
    }
    const kern = try kernelFor(.{ .kind = .nax, .bits = bits, .fused = up != null, .a = xd });
    const lim = try consts(.float32, &.{@bitCast(limit)});
    const u = up orelse w;
    return run(kern, &.{ x, w.packed_w, w.scales, u.packed_w, u.scales, idx, meta, lim }, .{
        .grid = .{ @divTrunc(n + 63, 64) * 128, @divTrunc(m + 63, 64), 1 },
        .group = .{ 128, 1, 1 },
        .shape = &.{ m, n },
        .dtype = xd,
    }, s);
}

/// kernels.py _gather_qmm_sorted_steel (no NAX): MLX's non-NAX gather_qmm_rhs tiles with the TQ loader.
fn steelSorted(s: mlx.mlx_stream, x: mlx.mlx_array, w: Proj, idx: mlx.mlx_array, meta: mlx.mlx_array, bits: u32) !mlx.mlx_array {
    const m = mlx.getShape(x)[0];
    const n = mlx.getShape(w.packed_w)[1];
    const xd = mlx.mlx_array_dtype(x);
    const kern = try kernelFor(.{ .kind = .steel, .bits = bits, .a = xd });
    return run(kern, &.{ x, w.packed_w, w.scales, idx, meta }, .{
        .grid = .{ @divTrunc(n + 31, 32) * 64, @divTrunc(m + 15, 16), 1 },
        .group = .{ 64, 1, 1 },
        .shape = &.{ m, n },
        .dtype = xd,
    }, s);
}

/// The steel arm's activation, on the host graph as gather_qmm_sorted runs it: f32 g and u,
/// clamped at `limit` > 0, (g * sigmoid(g)) * u, back to `out`. Takes ownership of `up`.
fn steelSwiglu(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, limit: f32, out: mlx.mlx_dtype) !mlx.mlx_array {
    defer free(up);
    const uf = try astype(up, .float32, s);
    defer free(uf);
    const gf = try astype(gate, .float32, s);
    defer free(gf);
    var gc = mlx.mlx_array_new();
    defer free(gc);
    var uc = mlx.mlx_array_new();
    defer free(uc);
    var g = gf;
    var u = uf;
    if (limit > 0) {
        const hi = mlx.mlx_array_new_float(limit);
        defer free(hi);
        const lo = mlx.mlx_array_new_float(-limit);
        defer free(lo);
        try mlx.check(mlx.mlx_minimum(&gc, gf, hi, s));
        try mlx.check(mlx.mlx_clip(&uc, uf, lo, hi, s));
        g = gc;
        u = uc;
    }
    var sig = mlx.mlx_array_new();
    defer free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var act = mlx.mlx_array_new();
    defer free(act);
    try mlx.check(mlx.mlx_multiply(&act, g, sig, s));
    var prod = mlx.mlx_array_new();
    defer free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, act, u, s));
    return astype(prod, out, s);
}

/// kernels.py weighted_unsort: y [T*k, D] sorted expert rows, inv [T*k] (the sort's inverse), wts
/// [T*k] f32 -> [T, D] in y's dtype, fp32 accumulation over a token's k rows in routing order.
fn weightedUnsort(s: mlx.mlx_stream, y: mlx.mlx_array, inv: mlx.mlx_array, wts: mlx.mlx_array, tokens: c_int, k: c_int) !mlx.mlx_array {
    const d = mlx.getShape(y)[1];
    if (@rem(d, 4) != 0) return error.Jangtq2Shape;
    const yd = mlx.mlx_array_dtype(y);
    const kern = try kernelFor(.{ .kind = .unsort, .a = yd });
    return run(kern, &.{ y, inv, wts }, .{
        .grid = .{ @divExact(d, 4), tokens, 1 },
        .group = .{ @min(256, @divExact(d, 4)), 1, 1 },
        .shape = &.{ tokens, d },
        .dtype = yd,
        .template = &.{ .{ .name = "D", .value = d }, .{ .name = "KK", .value = k } },
    }, s);
}

// ------------------------------------------------------------------ kernels

const Kind = enum { h32, qmv, wdown, nax, steel, unsort };
/// One compiled variant: `a`/`b` are the dtypes its source names (see kernelFor).
const KernelKey = struct { kind: Kind, bits: u32 = 0, fused: bool = false, a: mlx.mlx_dtype = .float32, b: mlx.mlx_dtype = .float32, mode: Mode = .al };
var kernels: std.AutoHashMapUnmanaged(KernelKey, mlx.mlx_fast_metal_kernel) = .empty;
var nax_header: ?[:0]const u8 = null;
var steel_header: ?[:0]const u8 = null;

fn typeName(d: mlx.mlx_dtype) ?[]const u8 {
    return switch (d) {
        .bfloat16 => "bfloat16_t",
        .float16 => "half",
        .float32 => "float",
        else => null,
    };
}

fn kernelFor(key: KernelKey) !mlx.mlx_fast_metal_kernel {
    if (kernels.get(key)) |k| return k;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ta = typeName(key.a) orelse return error.Jangtq2Dtype;
    const tb = typeName(key.b) orelse return error.Jangtq2Dtype;
    const role = if (key.fused) "fused" else "single";
    const k = switch (key.kind) {
        .h32 => try newKernel(try a.printSentinel("msv_jangtq2_h32_{s}_{s}", .{ ta, tb }, 0), &.{ "x", "meta" }, "out", try h32Source(a, tb), ""),
        .qmv => try newKernel(
            try a.printSentinel("msv_jangtq2_qmv_b{d}_{s}_{t}", .{ key.bits, ta, key.mode }, 0),
            &.{ "x", "wg", "sg", "wu", "su", "idx", "meta", "lim" },
            "out",
            try qmvSource(a, key.bits, key.mode),
            CODEBOOK,
        ),
        .wdown => try newKernel(
            try a.printSentinel("msv_jangtq2_wdown_b{d}_{s}_{s}_{t}", .{ key.bits, ta, tb, key.mode }, 0),
            &.{ "x", "wg", "sg", "idx", "wts", "meta" },
            "out",
            try wdownSource(a, key.bits, ta, key.mode),
            CODEBOOK,
        ),
        .nax => try newKernel(
            try a.printSentinel("msv_jangtq2_nax_b{d}_{s}_{s}", .{ key.bits, role, ta }, 0),
            &.{ "x", "wg", "sg", "wu", "su", "indices", "meta", "lim" },
            "y",
            try naxSource(a, key.bits, key.fused, ta),
            try sortedHeader(.nax),
        ),
        .steel => try newKernel(
            try a.printSentinel("msv_jangtq2_steel_b{d}_{s}", .{ key.bits, ta }, 0),
            &.{ "x", "wg", "sg", "indices", "meta" },
            "y",
            try steelSource(a, key.bits, ta),
            try sortedHeader(.steel),
        ),
        .unsort => try newKernel(try a.printSentinel("msv_jangtq2_unsort_{s}", .{ta}, 0), &.{ "Y", "INV", "S" }, "OUT", try unsortSource(a, ta), ""),
    };
    errdefer _ = mlx.mlx_fast_metal_kernel_free(k);
    try kernels.put(std.heap.c_allocator, key, k);
    return k;
}

fn newKernel(name: [:0]const u8, ins: []const [*:0]const u8, out: [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const outs = [_][*:0]const u8{out};
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, source.ptr, header.ptr, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const Launch = struct {
    grid: [3]c_int,
    group: [3]c_int,
    shape: []const c_int,
    dtype: mlx.mlx_dtype,
    template: []const struct { name: [*:0]const u8, value: c_int } = &.{},
};

fn run(kern: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, l: Launch, s: mlx.mlx_stream) !mlx.mlx_array {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, l.shape.ptr, l.shape.len, l.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, l.group[0], l.group[1], l.group[2]));
    for (l.template) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.value));
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kern, in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outs) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

/// kernels.py _consts: a small immutable input, one array per distinct value list (`vals` are the
/// raw 32-bit words, f32 bits for .float32).
const ConstKey = struct { dtype: mlx.mlx_dtype, len: usize, vals: [8]u32 };
var const_arrays: std.AutoHashMapUnmanaged(ConstKey, mlx.mlx_array) = .empty;

fn consts(dtype: mlx.mlx_dtype, vals: []const u32) !mlx.mlx_array {
    var key: ConstKey = .{ .dtype = dtype, .len = vals.len, .vals = @splat(0) };
    @memcpy(key.vals[0..vals.len], vals);
    if (const_arrays.get(key)) |arr| return arr;
    const arr = mlx.mlx_array_new_data(vals.ptr, &[_]c_int{@intCast(vals.len)}, 1, dtype);
    errdefer free(arr);
    try const_arrays.put(std.heap.c_allocator, key, arr);
    return arr;
}

// ------------------------------------------------------------------ Metal sources (kernels.py, rotate=False)

/// kernels.py _qdot: accumulate dot(xt[0..15], level(q)) over one row's 16-value lane chunk at `wr`.
fn qdot(a: Allocator, bits: u32, wr: []const u8, acc: []const u8) ![]u8 {
    const mask = (@as(u32, 1) << @intCast(bits)) - 1;
    return switch (bits) {
        2, 4, 8 => a.print(
            \\
            \\        {{ {[u]s} for (uint j = 0; j < {[nw]d}u; j++) {{ uint ww = {[wr]s}[j];
            \\            {[u]s} for (uint t = 0; t < {[per]d}u; t++)
            \\              {[acc]s} = fma(xt[j * {[per]d}u + t], tq_level<{[bits]d}>((ww >> ({[bits]d}u * t)) & {[mask]d}u), {[acc]s}); }} }}
        , .{ .u = UNROLL, .nw = bits / 2, .wr = wr, .per = 32 / bits, .acc = acc, .bits = bits, .mask = mask }),
        6 => a.print(
            \\
            \\        {{ uint w0 = {[wr]s}[0], w1 = {[wr]s}[1], w2 = {[wr]s}[2];
            \\          ulong lo = ulong(w0) | (ulong(w1) << 32); ulong hi = ulong(w1) | (ulong(w2) << 32);
            \\          {[u]s} for (uint t = 0; t < 16u; t++) {{
            \\            uint bp = 6u * t;
            \\            uint q = bp < 32u ? uint((lo >> bp) & {[mask]d}ul) : uint((hi >> (bp - 32u)) & {[mask]d}ul);
            \\            {[acc]s} = fma(xt[t], tq_level<6>(q), {[acc]s}); }} }}
        , .{ .u = UNROLL, .wr = wr, .acc = acc, .mask = mask }),
        3 => a.print(
            \\
            \\        {{ uint s0 = {[wr]s}[0], s1 = {[wr]s}[1], s2 = {[wr]s}[2];
            \\          uint g0 = s0 | ((s1 & 0xffu) << 16); uint g1 = (s1 >> 8) | (s2 << 8);
            \\          {[u]s} for (uint t = 0; t < 8u; t++) {[acc]s} = fma(xt[t], tq_level<3>((g0 >> (3u * t)) & 7u), {[acc]s});
            \\          {[u]s} for (uint t = 0; t < 8u; t++) {[acc]s} = fma(xt[8u + t], tq_level<3>((g1 >> (3u * t)) & 7u), {[acc]s}); }}
        , .{ .u = UNROLL, .wr = wr, .acc = acc }),
        else => error.Jangtq2Bits,
    };
}

/// kernels.py _ptype: the element type packed weights are addressed by (alignment-provable), and its bytes.
fn ptype(bits: u32) struct { name: []const u8, size: u32 } {
    return if (bits == 3) .{ .name = "uint16_t", .size = 2 } else .{ .name = "uint32_t", .size = 4 };
}

/// kernels.py _wptr: typed pointer to row0's lane chunk of expert `e`.
fn wptr(a: Allocator, bits: u32, base: []const u8, e: []const u8) ![]u8 {
    const p = ptype(bits);
    const cast = if (p.size == 4) "" else "(const device uint16_t*)";
    return a.print("{s}{s} + ((size_t){s} * N + row0) * (RB / {d}u) + lane * (LB / {d}u)", .{ cast, base, e, p.size, p.size });
}

/// kernels.py _prologue.
fn prologue(a: Allocator, bits: u32) ![]u8 {
    return a.print(
        \\
        \\    const uint RB = K * {[bits]d}u / 8u;       // bytes per row (multiple of 4: K % 32 == 0)
        \\    const uint LB = {[lb]d}u;             // bytes per lane chunk (16 values)
        \\    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
        \\    uint row0 = threadgroup_position_in_grid.y * {[tg]d}u + sgi * {[rps]d}u;
    , .{ .bits = bits, .lb = 2 * bits, .tg = NSG * RPS, .rps = RPS });
}

/// kernels.py _load_x: aligned blocks load unconditionally (vectorizable), the guarded form zeroes
/// lanes past K.
fn loadX(a: Allocator, aligned: bool) ![]u8 {
    if (aligned) return a.print(
        \\
        \\      const bool active = true;
        \\      float xt[{[v]d}];
        \\      {[u]s} for (uint i = 0; i < {[v]d}u; i++) xt[i] = float(xp[i]);
    , .{ .v = VPT, .u = UNROLL });
    return a.print(
        \\
        \\      bool active = (k0 + lane * {[v]d}u) < K;
        \\      float xt[{[v]d}];
        \\      {[u]s} for (uint i = 0; i < {[v]d}u; i++) xt[i] = active ? float(xp[i]) : 0.0f;
    , .{ .v = VPT, .u = UNROLL });
}

/// kernels.py _k_loop: `body` over whole blocks, `tail` (rows guarded on `active`) over the last
/// partial one.
fn kLoop(a: Allocator, mode: Mode, body: []const u8, tail: []const u8, advance: []const u8) ![]u8 {
    return switch (mode) {
        .al => a.print("for (uint k0 = 0; k0 < K; k0 += {d}u) {{ {s} {s} {s} }}", .{ BLOCK, try loadX(a, true), body, advance }),
        .tl => a.print(
            "uint Kmain = K - (K % {[b]d}u);\n" ++
                "    for (uint k0 = 0; k0 < Kmain; k0 += {[b]d}u) {{ {[la]s} {[body]s} {[adv]s} }}\n" ++
                "    if (Kmain < K) {{ uint k0 = Kmain; {[lg]s} {[tail]s} }}",
            .{ .b = BLOCK, .la = try loadX(a, true), .body = body, .adv = advance, .lg = try loadX(a, false), .tail = tail },
        ),
    };
}

/// kernels.py _qmv_rows, gate and up.
fn qmvRows(a: Allocator, bits: u32, guard: []const u8) ![]u8 {
    const p = ptype(bits);
    return a.print(
        \\
        \\      {[u]s} for (uint r = 0; r < {[rps]d}u; r++) {{
        \\        {[guard]s}
        \\        const device {[pt]s}* wr = wp + r * (RB / {[sz]d}u); float a = 0.0f;
        \\        {[gate]s}
        \\        accg[r] += a;
        \\        const device {[pt]s}* ur = up + r * (RB / {[sz]d}u); float b2 = 0.0f;
        \\        {[up]s}
        \\        accu[r] += b2;
        \\      }}
    , .{ .u = UNROLL, .rps = RPS, .guard = guard, .pt = p.name, .sz = p.size, .gate = try qdot(a, bits, "wr", "a"), .up = try qdot(a, bits, "ur", "b2") });
}

/// kernels.py _qmv_kernel_mode, fused.
fn qmvSource(a: Allocator, bits: u32, mode: Mode) ![:0]u8 {
    const p = ptype(bits);
    const adv = BLOCK * bits / 8 / p.size;
    const loop = try kLoop(a, mode, try qmvRows(a, bits, ""), try qmvRows(a, bits, "if (!active) continue;"), try a.print("wp += {d}u; up += {d}u; xp += {d}u;", .{ adv, adv, BLOCK }));
    const store = try a.print(
        \\
        \\    float L = lim[0];
        \\    {[u]s} for (uint r = 0; r < {[rps]d}u; r++) {{
        \\      float g = simd_sum(accg[r]); float u = simd_sum(accu[r]);
        \\      if (lane == 0 && row0 + r < N) {{
        \\        g *= float(sg[(size_t)e * N + row0 + r]); u *= float(su[(size_t)e * N + row0 + r]);
        \\        if (L > 0.0f) {{ g = metal::min(g, L); u = metal::clamp(u, -L, L); }}
        \\        out[(size_t)disp * N + row0 + r] = (g / (1.0f + metal::fast::exp(-g))) * u;
        \\      }}
        \\    }}
    , .{ .u = UNROLL, .rps = RPS });
    return a.printSentinel(
        \\
        \\    uint K = meta[0], N = meta[1], xdiv = meta[2];
        \\    {[pro]s}
        \\    uint disp = threadgroup_position_in_grid.z;
        \\    uint e = idx[disp];
        \\    const device {[pt]s}* wp = {[wp]s};
        \\    const device {[pt]s}* up = {[up]s};
        \\    auto xp = x + (size_t)(disp / xdiv) * K + lane * {[v]d}u;
        \\    float accg[{[rps]d}]; float accu[{[rps]d}];
        \\    {[u]s} for (uint r = 0; r < {[rps]d}u; r++) {{ accg[r] = 0.0f; accu[r] = 0.0f; }}
        \\    {[loop]s}
        \\    {[store]s}
        \\
    , .{
        .pro = try prologue(a, bits),
        .pt = p.name,
        .wp = try wptr(a, bits, "wg", "e"),
        .up = try wptr(a, bits, "wu", "e"),
        .v = VPT,
        .rps = RPS,
        .u = UNROLL,
        .loop = loop,
        .store = store,
    }, 0);
}

/// kernels.py _wdown_rows.
fn wdownRows(a: Allocator, bits: u32, guard: []const u8) ![]u8 {
    const p = ptype(bits);
    return a.print(
        \\
        \\        {[u]s} for (uint r = 0; r < {[rps]d}u; r++) {{
        \\          {[guard]s}
        \\          const device {[pt]s}* wr = wp + r * (RB / {[sz]d}u); float a = 0.0f;
        \\          {[dot]s}
        \\          acc[r] += a;
        \\        }}
    , .{ .u = UNROLL, .rps = RPS, .guard = guard, .pt = p.name, .sz = p.size, .dot = try qdot(a, bits, "wr", "a") });
}

/// kernels.py _qmv_weighted_down_kernel: y[t, r] = sum_k w[t,k] * scale[e_k, r] * dot(h[t,k,:], level(q[e_k, r, :])).
fn wdownSource(a: Allocator, bits: u32, tname: []const u8, mode: Mode) ![:0]u8 {
    const p = ptype(bits);
    const loop = try kLoop(a, mode, try wdownRows(a, bits, ""), try wdownRows(a, bits, "if (!active) continue;"), try a.print("wp += {d}u; xp += {d}u;", .{ BLOCK * bits / 8 / p.size, BLOCK }));
    return a.printSentinel(
        \\
        \\    uint K = meta[0], N = meta[1], KT = meta[2];
        \\    {[pro]s}
        \\    uint t = threadgroup_position_in_grid.z;
        \\    float out_acc[{[rps]d}]; {[u]s} for (uint r = 0; r < {[rps]d}u; r++) out_acc[r] = 0.0f;
        \\    for (uint kk = 0; kk < KT; kk++) {{
        \\      uint disp = t * KT + kk;
        \\      uint e = idx[disp];
        \\      float wk = float(wts[disp]);
        \\      const device {[pt]s}* wp = {[wp]s};
        \\      auto xp = x + (size_t)disp * K + lane * {[v]d}u;
        \\      float acc[{[rps]d}]; {[u]s} for (uint r = 0; r < {[rps]d}u; r++) acc[r] = 0.0f;
        \\      {[loop]s}
        \\      {[u]s} for (uint r = 0; r < {[rps]d}u; r++) {{
        \\        float sres = simd_sum(acc[r]);
        \\        if (row0 + r < N) out_acc[r] += wk * sres * float(sg[(size_t)e * N + row0 + r]);
        \\      }}
        \\    }}
        \\    {[u]s} for (uint r = 0; r < {[rps]d}u; r++)
        \\      if (lane == 0 && row0 + r < N) out[(size_t)t * N + row0 + r] = static_cast<{[t]s}>(out_acc[r]);
        \\
    , .{
        .pro = try prologue(a, bits),
        .rps = RPS,
        .u = UNROLL,
        .pt = p.name,
        .wp = try wptr(a, bits, "wg", "e"),
        .v = VPT,
        .loop = loop,
        .t = tname,
    }, 0);
}

/// kernels.py _h32_kernel: one simdgroup per 32-wide block, lane i holds x[block*32 + i]; five
/// butterfly stages across the lanes.
fn h32Source(a: Allocator, oname: []const u8) ![:0]u8 {
    return a.printSentinel(
        \\
        \\    uint K = meta[0];
        \\    uint lane = thread_index_in_simdgroup;
        \\    size_t off = (size_t)threadgroup_position_in_grid.z * K + (size_t)threadgroup_position_in_grid.y * 32u + lane;
        \\    float v = float(x[off]);
        \\    {[u]s} for (uint h = 1u; h < 32u; h <<= 1) {{
        \\      float o = simd_shuffle_xor(v, ushort(h));
        \\      v = (lane & h) ? (o - v) : (v + o);
        \\    }}
        \\    out[off] = static_cast<{[o]s}>(v * 0.17677669529663687f);
        \\
    , .{ .u = UNROLL, .o = oname }, 0);
}

/// kernels.py _nax_kernel.
fn naxSource(a: Allocator, bits: u32, fused: bool, tname: []const u8) ![:0]u8 {
    return a.printSentinel(
        \\
        \\  constexpr int BK_padded = (64 + 16 / sizeof({[t]s}));
        \\  threadgroup {[t]s} Wg[64 * BK_padded];
        \\  threadgroup {[t]s} Wu[{[wu]s} * BK_padded];
        \\  tq_gather_qmm_nax<{[t]s}, {[bits]d}, {[fused]s}>(
        \\      x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], lim[0], Wg, Wu,
        \\      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        \\
    , .{ .t = tname, .wu = if (fused) "64" else "1", .bits = bits, .fused = if (fused) "true" else "false" }, 0);
}

/// kernels.py _steel_kernel.
fn steelSource(a: Allocator, bits: u32, tname: []const u8) ![:0]u8 {
    return a.printSentinel(
        \\
        \\  constexpr int BK_padded = (32 + 16 / sizeof({[t]s}));
        \\  threadgroup {[t]s} Xs[16 * BK_padded];
        \\  threadgroup {[t]s} Ws[32 * BK_padded];
        \\  tq_gather_qmm_steel<{[t]s}, {[bits]d}>(x, wg, sg, indices, y, meta[0], meta[1], meta[2], Xs, Ws,
        \\      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        \\
    , .{ .t = tname, .bits = bits }, 0);
}

/// kernels.py _weighted_unsort_kernel, except the row bound: vMLX templates it (ROWS), a JIT per
/// prompt length; here it is the dispatch's own height. The accumulation is unchanged.
fn unsortSource(a: Allocator, tname: []const u8) ![:0]u8 {
    return a.printSentinel(
        \\
        \\const uint d4 = thread_position_in_grid.x;            // column quad
        \\const uint t = thread_position_in_grid.y;             // token
        \\if (d4 * 4u >= D || t >= threads_per_grid.y) return;
        \\float4 acc = float4(0.0f);
        \\for (uint j = 0; j < KK; ++j) {{
        \\  const uint row = INV[t * KK + j];
        \\  const float w = float(S[t * KK + j]);
        \\  const device {[t]s}* yr = Y + (size_t)row * D + d4 * 4u;
        \\  acc += w * float4(float(yr[0]), float(yr[1]), float(yr[2]), float(yr[3]));
        \\}}
        \\device {[t]s}* o = OUT + (size_t)t * D + d4 * 4u;
        \\o[0] = {[t]s}(acc.x); o[1] = {[t]s}(acc.y); o[2] = {[t]s}(acc.z); o[3] = {[t]s}(acc.w);
        \\
    , .{ .t = tname }, 0);
}

/// The sorted GEMMs' header (kernels.py _nax_header / _steel_header): MLX's own kernel headers
/// inlined, the codebook, the TQ tile loader and the GEMM. Built once.
fn sortedHeader(arm: Prefill) ![:0]const u8 {
    const slot = if (arm == .nax) &nax_header else &steel_header;
    if (slot.*) |h| return h;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const files: []const []const u8 = switch (arm) {
        .nax => &.{ "steel/gemm/gemm.h", "steel/gemm/nax.h", "steel/gemm/loader.h", "quantized_nax.h" },
        .steel => &.{ "steel/gemm/gemm.h", "quantized_utils.h", "quantized.h" },
    };
    const impl = switch (arm) {
        .nax => @embedFile("kernels/jangtq2_nax.metal"),
        .steel => @embedFile("kernels/jangtq2_steel.metal"),
    };
    const parts = [_][]const u8{ try mlx_headers.inlined(arena.allocator(), files), CODEBOOK, @embedFile("kernels/jangtq2_loader.metal"), impl };
    const h = try std.mem.concatWithSentinel(std.heap.c_allocator, u8, &parts, 0);
    slot.* = h;
    return h;
}

// ------------------------------------------------------------------ array helpers

fn free(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

fn dup(a: mlx.mlx_array) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_array_set(&out, a));
    return out;
}

fn reshape(a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_reshape(&out, a, shape.ptr, shape.len, s));
    return out;
}

fn astype(a: mlx.mlx_array, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, a, dtype, s));
    return out;
}

// ------------------------------------------------------------------ tests

const testing = std.testing;

fn requireMetal() !void {
    mlx.installErrorHandler();
    if (mlx.noGpuBackend() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
}

/// A kernel failure is its test's: name it and drop its latch, or the next test inherits it.
fn dropOwnLatch() void {
    var buf: [512]u8 = undefined;
    if (mlx.takeError(&buf)) |msg| std.debug.print("[jangtq2] mlx: {s}\n", .{msg});
}

fn zerosFor(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_zeros(&out, shape.ptr, shape.len, dtype, s));
    return out;
}

test "jangtq2 moe refuses malformed inputs and banks by name" {
    try requireMetal();
    const s = mlx.gpuStream();
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| free(a);
        owned.deinit(testing.allocator);
    }
    const z = struct {
        fn make(list: *std.ArrayList(mlx.mlx_array), shape: []const c_int, dtype: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            const a = try zerosFor(shape, dtype, st);
            errdefer free(a);
            try list.append(testing.allocator, a);
            return a;
        }
    }.make;
    // E=2, D=64, I=64: gate/up 4-bit, down 6-bit.
    const gw = try z(&owned, &.{ 2, 64, 8 }, .uint32, s);
    const sc = try z(&owned, &.{ 2, 64 }, .float16, s);
    const dw = try z(&owned, &.{ 2, 64, 12 }, .uint32, s);
    const bank: Bank = .{ .gate = .{ .packed_w = gw, .scales = sc }, .up = .{ .packed_w = gw, .scales = sc }, .down = .{ .packed_w = dw, .scales = sc }, .rotated = true };
    const x = try z(&owned, &.{ 1, 2, 64 }, .bfloat16, s);
    const inds = try z(&owned, &.{ 1, 2, 2 }, .uint32, s);
    const scores = try z(&owned, &.{ 1, 2, 2 }, .bfloat16, s);

    try testing.expectError(error.Jangtq2Dtype, moe(s, try z(&owned, &.{ 1, 2, 64 }, .int32, s), bank, inds, scores, 0));
    try testing.expectError(error.Jangtq2Dtype, moe(s, x, bank, scores, scores, 0));
    try testing.expectError(error.Jangtq2Dtype, moe(s, x, bank, inds, inds, 0));
    try testing.expectError(error.Jangtq2Shape, moe(s, x, bank, inds, try z(&owned, &.{ 1, 2, 3 }, .bfloat16, s), 0));
    try testing.expectError(error.Jangtq2Shape, moe(s, x, bank, try z(&owned, &.{ 2, 1, 2 }, .uint32, s), try z(&owned, &.{ 2, 1, 2 }, .bfloat16, s), 0));
    try testing.expectError(error.Jangtq2Shape, moe(s, try z(&owned, &.{ 1, 2, 48 }, .bfloat16, s), bank, inds, scores, 0));
    var bad = bank;
    bad.down.packed_w = try z(&owned, &.{ 2, 64, 10 }, .uint32, s); // 5 bits per code
    try testing.expectError(error.Jangtq2Bits, moe(s, x, bad, inds, scores, 0));
    bad = bank;
    bad.up.scales = try z(&owned, &.{ 2, 64 }, .bfloat16, s);
    try testing.expectError(error.Jangtq2Dtype, moe(s, x, bad, inds, scores, 0));
    bad = bank;
    bad.up.packed_w = dw; // gate and up must share one width
    try testing.expectError(error.Jangtq2Shape, moe(s, x, bad, inds, scores, 0));
    bad = bank;
    bad.down.scales = try z(&owned, &.{ 3, 64 }, .float16, s);
    try testing.expectError(error.Jangtq2Shape, moe(s, x, bad, inds, scores, 0));

    // Past the decode width: f32 cannot take the fused NAX tile, and the GEMMs need K % 64.
    const inds97 = try z(&owned, &.{ 97, 2 }, .uint32, s);
    const scores97 = try z(&owned, &.{ 97, 2 }, .float32, s);
    try testing.expectError(error.Jangtq2Dtype, moeOn(s, try z(&owned, &.{ 97, 64 }, .float32, s), bank, inds97, scores97, 0, .nax));
    const gw96 = try z(&owned, &.{ 2, 64, 12 }, .uint32, s);
    const dw96 = try z(&owned, &.{ 2, 96, 8 }, .uint32, s);
    const sc96 = try z(&owned, &.{ 2, 96 }, .float16, s);
    const bank96: Bank = .{ .gate = .{ .packed_w = gw96, .scales = sc }, .up = .{ .packed_w = gw96, .scales = sc }, .down = .{ .packed_w = dw96, .scales = sc96 }, .rotated = true };
    try testing.expectError(error.Jangtq2Shape, moeOn(s, try z(&owned, &.{ 97, 96 }, .bfloat16, s), bank96, inds97, scores97, 0, .nax));
}

/// contract.CUBIC_PARAMS: level(q) = u * (alpha + beta * u^2), u = q - (2^bits - 1) / 2.
fn levelRef(bits: u32, q: u32) f64 {
    const ab: [2]f64 = switch (bits) {
        2 => .{ 0.8929999999999999, 0.05065 },
        3 => .{ 0.47124999999999995, 0.011599999999999997 },
        4 => .{ 0.2405, 0.002100000000000002 },
        6 => .{ 0.10413533834586466, 0.0 },
        8 => .{ 0.030780075187969925, 0.0 },
        else => unreachable,
    };
    const u = @as(f64, @floatFromInt(q)) - @as(f64, @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1)) / 2.0;
    return u * (ab[0] + ab[1] * u * u);
}

/// A random [e, n, k] bank in the v2 layout: codes packed as format.pack_bitstream does (row-wise
/// LSB-first, value j at bit j * bits), float16 row scales.
const SynthProj = struct {
    codes: []u8,
    scales: []f16,
    words: []u32,
    n: usize,
    k: usize,
    bits: u32,

    fn init(a: Allocator, rng: std.Random, e: usize, n: usize, k: usize, bits: u32) !SynthProj {
        const codes = try a.alloc(u8, e * n * k);
        for (codes) |*c| c.* = @intCast(rng.uintLessThan(u32, @as(u32, 1) << @intCast(bits)));
        const scales = try a.alloc(f16, e * n);
        for (scales) |*v| v.* = @floatCast(0.03 + 0.02 * rng.float(f32));
        const wpr = k * bits / 32;
        const words = try a.alloc(u32, e * n * wpr);
        @memset(words, 0);
        for (0..e * n) |row| {
            const out = words[row * wpr ..][0..wpr];
            for (codes[row * k ..][0..k], 0..) |q, j| {
                const pos = j * bits;
                const off: u5 = @intCast(pos % 32);
                out[pos / 32] |= @as(u32, q) << off;
                if (@as(u32, off) + bits > 32) out[pos / 32 + 1] |= @as(u32, q) >> @intCast(32 - @as(u32, off));
            }
        }
        return .{ .codes = codes, .scales = scales, .words = words, .n = n, .k = k, .bits = bits };
    }

    fn weight(self: SynthProj, e: usize, row: usize, j: usize) f64 {
        const r = e * self.n + row;
        return @as(f64, self.scales[r]) * levelRef(self.bits, self.codes[r * self.k + j]);
    }

    fn proj(self: SynthProj, experts: usize) Proj {
        const wpr: c_int = @intCast(self.k * self.bits / 32);
        return .{
            .packed_w = mlx.mlx_array_new_data(self.words.ptr, &[_]c_int{ @intCast(experts), @intCast(self.n), wpr }, 3, .uint32),
            .scales = mlx.mlx_array_new_data(self.scales.ptr, &[_]c_int{ @intCast(experts), @intCast(self.n) }, 2, .float16),
        };
    }
};

/// Blockwise normalized Walsh-Hadamard-32 (natural order), in place.
fn h32Ref(v: []f64) void {
    var b: usize = 0;
    while (b < v.len) : (b += 32) {
        const blk = v[b..][0..32];
        var h: usize = 1;
        while (h < 32) : (h <<= 1) {
            for (0..32) |i| if (i & h == 0) {
                const lo = blk[i];
                const hi = blk[i + h];
                blk[i] = lo + hi;
                blk[i + h] = lo - hi;
            };
        }
        for (blk) |*x| x.* /= @sqrt(32.0);
    }
}

fn bf16Round(v: f32) f32 {
    const b: u32 = @bitCast(v);
    return @bitCast((b +% 0x7fff +% ((b >> 16) & 1)) & 0xffff0000);
}

/// The routed sum in float64 from the format definition: gate/up/SwiGLU (clamped at `limit` > 0)
/// per routed expert, H32 before every projection when `rotated`, router-weighted down rows.
fn moeRef(a: Allocator, gate: SynthProj, up: SynthProj, down: SynthProj, x: []const f32, inds: []const i32, scores: []const f32, k: usize, rotated: bool, limit: f64) ![]f64 {
    const d = gate.k;
    const i_dim = gate.n;
    const tokens = x.len / d;
    const out = try a.alloc(f64, tokens * d);
    @memset(out, 0);
    const xr = try a.alloc(f64, d);
    defer a.free(xr);
    const h = try a.alloc(f64, i_dim);
    defer a.free(h);
    for (0..tokens) |t| {
        for (xr, x[t * d ..][0..d]) |*dst, v| dst.* = v;
        if (rotated) h32Ref(xr);
        for (0..k) |j| {
            const e: usize = @intCast(inds[t * k + j]);
            for (h, 0..) |*hv, row| {
                var g: f64 = 0;
                var u: f64 = 0;
                for (xr, 0..) |xv, c| {
                    g += gate.weight(e, row, c) * xv;
                    u += up.weight(e, row, c) * xv;
                }
                if (limit > 0) {
                    g = @min(g, limit);
                    u = std.math.clamp(u, -limit, limit);
                }
                hv.* = g / (1.0 + @exp(-g)) * u;
            }
            if (rotated) h32Ref(h);
            for (out[t * d ..][0..d], 0..) |*o, row| {
                var y: f64 = 0;
                for (h, 0..) |hv, c| y += down.weight(e, row, c) * hv;
                o.* += scores[t * k + j] * y;
            }
        }
    }
    return out;
}

fn hostF32(a: Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    const f = try astype(arr, .float32, s);
    defer free(f);
    var c = mlx.mlx_array_new();
    defer free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const p = mlx.mlx_array_data_float32(c) orelse return error.MlxError;
    return a.dupe(f32, p[0..n]);
}

/// sqrt(sum (got - ref)^2 / sum ref^2).
fn relRms(got: []const f32, ref: []const f64) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, ref) |g, r| {
        num += (g - r) * (g - r);
        den += r * r;
    }
    return @sqrt(num / den);
}

test "jangtq2 moe matches the float64 format definition on synthetic banks, decode and both prefill arms" {
    try requireMetal();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7a2);
    const rng = prng.random();
    const E = 8;
    const D = 512;
    const I = 64;
    const K = 3;
    // D = 512 gives the unguarded gate/up decode loop, I = 64 the tail-split down loop.
    const combos = [_]struct { gu: u32, dn: u32, rotated: bool, limit: f32 }{
        .{ .gu = 4, .dn = 6, .rotated = true, .limit = 0 },
        .{ .gu = 2, .dn = 8, .rotated = false, .limit = 0 },
        .{ .gu = 3, .dn = 4, .rotated = true, .limit = 0.75 },
    };
    for (combos) |c| {
        const gate = try SynthProj.init(a, rng, E, I, D, c.gu);
        const up = try SynthProj.init(a, rng, E, I, D, c.gu);
        const down = try SynthProj.init(a, rng, E, D, I, c.dn);
        const bank: Bank = .{ .gate = gate.proj(E), .up = up.proj(E), .down = down.proj(E), .rotated = c.rotated };
        defer for ([_]Proj{ bank.gate, bank.up, bank.down }) |p| {
            free(p.packed_w);
            free(p.scales);
        };
        for ([_]usize{ 5, 100 }) |tokens| {
            const xs = try a.alloc(u16, tokens * D);
            const xf = try a.alloc(f32, tokens * D);
            for (xs, xf) |*b, *f| {
                f.* = bf16Round(rng.floatNorm(f32));
                b.* = @intCast(@as(u32, @bitCast(f.*)) >> 16);
            }
            const ids = try a.alloc(i32, tokens * K);
            for (0..tokens) |t| {
                var picked: [E]bool = @splat(false);
                for (0..K) |j| {
                    var e = rng.uintLessThan(usize, E);
                    while (picked[e]) e = rng.uintLessThan(usize, E);
                    picked[e] = true;
                    ids[t * K + j] = @intCast(e);
                }
            }
            const sc = try a.alloc(u16, tokens * K);
            const scf = try a.alloc(f32, tokens * K);
            for (sc, scf) |*b, *f| {
                f.* = bf16Round(0.1 + 0.3 * rng.float(f32));
                b.* = @intCast(@as(u32, @bitCast(f.*)) >> 16);
            }
            const ref = try moeRef(a, gate, up, down, xf, ids, scf, K, c.rotated, c.limit);
            const x = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ 1, @intCast(tokens), D }, 3, .bfloat16);
            defer free(x);
            const inds = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(tokens), K }, 3, .int32);
            defer free(inds);
            const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ 1, @intCast(tokens), K }, 3, .bfloat16);
            defer free(scores);
            const gather = tokens <= DECODE_MAX_TOKENS;
            const arms: []const Prefill = if (gather or !@import("transformer.zig").naxAvailable()) &.{.steel} else &.{ .nax, .steel };
            for (arms) |arm| {
                const y = try moeOn(s, x, bank, inds, scores, c.limit, arm);
                defer free(y);
                try testing.expectEqualSlices(c_int, &.{ 1, @intCast(tokens), D }, mlx.getShape(y));
                try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
                const err = relRms(try hostF32(a, y, s), ref);
                // Decode rounds x's rotation and the output to bf16; prefill also its GEMM operands and h.
                const bar: f64 = if (gather) 6e-3 else 1.2e-2;
                if (!(err <= bar)) std.debug.print("[jangtq2] bits {d}/{d} rotated={any} limit={d} T={d} arm={t}: rel rms {e} > {e}\n", .{ c.gu, c.dn, c.rotated, c.limit, tokens, arm, err, bar });
                try testing.expect(err <= bar);
            }
            // f32 activations through the public entry: on an M5 the prefill must leave the NAX tile.
            const x32 = try astype(x, .float32, s);
            defer free(x32);
            const y32 = try moe(s, x32, bank, inds, scores, c.limit);
            defer free(y32);
            try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y32));
            const err32 = relRms(try hostF32(a, y32, s), ref);
            const bar32: f64 = if (gather) 6e-3 else 1.2e-2;
            if (!(err32 <= bar32)) std.debug.print("[jangtq2] f32 x bits {d}/{d} T={d}: rel rms {e} > {e}\n", .{ c.gu, c.dn, tokens, err32, bar32 });
            try testing.expect(err32 <= bar32);
        }
    }
}

fn readJson(a: Allocator, io: std.Io, path: []const u8) !std.json.Parsed(std.json.Value) {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    defer a.free(bytes);
    return std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
}

/// A safetensors file's tensors, read lazily on the CPU stream.
fn loadTensors(a: Allocator, path: []const u8) !mlx.mlx_map_string_to_array {
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    var map = mlx.mlx_map_string_to_array_new();
    errdefer _ = mlx.mlx_map_string_to_array_free(map);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&map, &meta, z.ptr, cpu));
    return map;
}

fn tensor(a: Allocator, map: mlx.mlx_map_string_to_array, name: []const u8) !mlx.mlx_array {
    const z = try a.dupeSentinel(u8, name, 0);
    defer a.free(z);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    if (mlx.mlx_map_string_to_array_get(&out, map, z.ptr) != 0) {
        std.debug.print("[jangtq2-fixture] missing tensor {s}\n", .{name});
        return error.FixtureTensorMissing;
    }
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// One layer's routed bank read from the bundle, with the widths and rotation its config declares.
const LayerBank = struct {
    arrays: [6]mlx.mlx_array,
    bank: Bank,
    bits_gu: u32,
    bits_dn: u32,

    fn load(a: Allocator, io: std.Io, bundle: []const u8, layer: i64) !LayerBank {
        const index = try readJson(a, io, try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{bundle}));
        defer index.deinit();
        const config = try readJson(a, io, try std.fmt.allocPrint(a, "{s}/config.json", .{bundle}));
        defer config.deinit();
        const weight_map = index.value.object.get("weight_map").?.object;
        const quant = config.value.object.get("quantization").?.object;
        var out: LayerBank = undefined;
        var n: usize = 0;
        errdefer for (out.arrays[0..n]) |x| free(x);
        var bits: [3]i64 = undefined;
        var rotated = true;
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |p, pi| {
            const entry = quant.get(try std.fmt.allocPrint(a, "model.layers.{d}.mlp.switch_mlp.{s}_proj", .{ layer, p })).?.object;
            if (!std.mem.eql(u8, entry.get("mode").?.string, "jangtq2")) return error.FixtureNotJangtq2;
            bits[pi] = entry.get("bits").?.integer;
            rotated = rotated and std.mem.eql(u8, entry.get("rotation").?.string, "hadamard32");
            for ([_][]const u8{ "tq2_packed", "tq2_scales" }) |leaf| {
                const name = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}", .{ layer, p, leaf });
                const shard = weight_map.get(name).?.string;
                const map = try loadTensors(a, try std.fmt.allocPrint(a, "{s}/{s}", .{ bundle, shard }));
                defer _ = mlx.mlx_map_string_to_array_free(map);
                out.arrays[n] = try tensor(a, map, name);
                n += 1;
            }
        }
        const p = out.arrays;
        out.bank = .{ .gate = .{ .packed_w = p[0], .scales = p[1] }, .up = .{ .packed_w = p[2], .scales = p[3] }, .down = .{ .packed_w = p[4], .scales = p[5] }, .rotated = rotated };
        const d = mlx.getShape(p[4])[1];
        const i = mlx.getShape(p[0])[1];
        out.bits_gu = try bitsOf(mlx.getShape(p[0])[2], d);
        out.bits_dn = try bitsOf(mlx.getShape(p[4])[2], i);
        // The config's declared widths are the ones the packed shapes imply.
        try testing.expectEqual(bits[0], @as(i64, out.bits_gu));
        try testing.expectEqual(bits[1], @as(i64, out.bits_gu));
        try testing.expectEqual(bits[2], @as(i64, out.bits_dn));
        return out;
    }

    fn deinit(self: LayerBank) void {
        for (self.arrays) |x| free(x);
    }
};

/// Bitwise equality: same dtype, shape and bytes. A mismatch prints how many elements differ.
fn sameBits(case: []const u8, what: []const u8, got: mlx.mlx_array, want: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    defer free(got);
    const gd = mlx.mlx_array_dtype(got);
    if (gd != mlx.mlx_array_dtype(want) or !std.mem.eql(c_int, mlx.getShape(got), mlx.getShape(want))) {
        std.debug.print("[jangtq2-fixture] {s} {s}: got {t} {any}, want {t} {any}\n", .{ case, what, gd, mlx.getShape(got), mlx.mlx_array_dtype(want), mlx.getShape(want) });
        return false;
    }
    var gc = mlx.mlx_array_new();
    defer free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    var wc = mlx.mlx_array_new();
    defer free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    try mlx.check(mlx.mlx_array_eval(gc));
    try mlx.check(mlx.mlx_array_eval(wc));
    const size = mlx.mlx_array_size(gc);
    const item = mlx.mlx_array_itemsize(gc);
    const gb = (mlx.mlx_array_data_uint8(gc) orelse return error.MlxError)[0 .. size * item];
    const wb = (mlx.mlx_array_data_uint8(wc) orelse return error.MlxError)[0 .. size * item];
    if (std.mem.eql(u8, gb, wb)) return true;
    var differ: usize = 0;
    var worst: f64 = 0;
    for (0..size) |e| {
        const g = gb[e * item ..][0..item];
        const w = wb[e * item ..][0..item];
        if (std.mem.eql(u8, g, w)) continue;
        differ += 1;
        const gv = elementF64(gd, g);
        const wv = elementF64(gd, w);
        worst = @max(worst, @abs(gv - wv));
    }
    std.debug.print("[jangtq2-fixture] {s} {s}: MISMATCH {d}/{d} elements differ, max |diff| {e}\n", .{ case, what, differ, size, worst });
    return false;
}

fn elementF64(dtype: mlx.mlx_dtype, b: []const u8) f64 {
    return switch (dtype) {
        .bfloat16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[0..2], .little)) << 16)),
        .float16 => @as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little))),
        .float32 => @as(f32, @bitCast(std.mem.readInt(u32, b[0..4], .little))),
        .uint32 => @floatFromInt(std.mem.readInt(u32, b[0..4], .little)),
        else => std.math.nan(f64),
    };
}

/// A projection over its first `words` packed words per row (its leading inputs), sharing the row
/// scales; the caller frees `packed_w`.
fn leadingWords(p: Proj, words: c_int, s: mlx.mlx_stream) !Proj {
    const sh = mlx.getShape(p.packed_w);
    var w = mlx.mlx_array_new();
    errdefer free(w);
    try mlx.check(mlx.mlx_slice(&w, p.packed_w, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], sh[1], words }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return .{ .packed_w = w, .scales = p.scales };
}

/// A fixture case's arrays, read on first use and held until the case ends.
const CaseArrays = struct {
    a: Allocator,
    map: mlx.mlx_map_string_to_array,
    held: std.StringHashMapUnmanaged(mlx.mlx_array) = .empty,

    fn get(self: *CaseArrays, key: []const u8) !mlx.mlx_array {
        if (self.held.get(key)) |v| return v;
        const v = try tensor(self.a, self.map, key);
        errdefer free(v);
        try self.held.put(self.a, key, v);
        return v;
    }

    fn deinit(self: *CaseArrays) void {
        var it = self.held.valueIterator();
        while (it.next()) |v| free(v.*);
        _ = mlx.mlx_map_string_to_array_free(self.map);
    }
};

const Tally = struct {
    case: []const u8,
    s: mlx.mlx_stream,
    checks: usize = 0,
    fails: usize = 0,

    /// Takes ownership of `got`.
    fn expect(self: *Tally, what: []const u8, got: mlx.mlx_array, want: mlx.mlx_array) !void {
        self.checks += 1;
        if (!try sameBits(self.case, what, got, want, self.s)) self.fails += 1;
    }
};

fn flatAs(s: mlx.mlx_stream, arr: mlx.mlx_array, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const r = try reshape(arr, &.{@intCast(mlx.mlx_array_size(arr))}, s);
    defer free(r);
    return astype(r, dtype, s);
}

fn takeRows(s: mlx.mlx_stream, a: mlx.mlx_array, rows: mlx.mlx_array) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_take_axis(&out, a, rows, 0, s));
    return out;
}

/// Runs one fixture case's checks; returns how many arrays differ. Kinds: decode and prefill (every
/// intermediate of routed()), routed, tl_fused.
fn fixtureCase(a: Allocator, dir: []const u8, case: std.json.ObjectMap, lb: LayerBank, arm: Prefill, s: mlx.mlx_stream) !usize {
    const name = case.get("name").?.string;
    const kind = case.get("kind").?.string;
    var c: CaseArrays = .{ .a = a, .map = try loadTensors(a, try std.fmt.allocPrint(a, "{s}/{s}.safetensors", .{ dir, name })) };
    defer c.deinit();
    var t: Tally = .{ .case = name, .s = s };
    const out = try c.get("out");
    const x = try c.get("x");
    const xd = mlx.mlx_array_dtype(x);
    const is = std.mem.eql;
    if (is(u8, kind, "decode") or is(u8, kind, "prefill") or is(u8, kind, "routed")) {
        const inds = try c.get("inds");
        const scores = try c.get("scores");
        const tokens = mlx.getShape(x)[0];
        const k = mlx.getShape(inds)[1];
        const idx = try flatAs(s, inds, .uint32);
        defer free(idx);
        const wts = try flatAs(s, scores, .float32);
        defer free(wts);
        if (is(u8, kind, "decode")) {
            try t.expect("xr", try h32Rows(s, x, xd), try c.get("xr"));
            try t.expect("h", try gatherQmv(s, try c.get("xr"), lb.bank.gate, lb.bank.up, idx, lb.bits_gu, 0), try c.get("h"));
            try t.expect("hr", try h32Rows(s, try c.get("h"), .float32), try c.get("hr"));
            try t.expect("out", try gatherQmvWeightedDown(s, try c.get("hr"), lb.bank.down, idx, wts, tokens, k, lb.bits_dn, xd), out);
        } else if (is(u8, kind, "prefill")) {
            var r = try @import("transformer.zig").sortRoutes(s, idx, k);
            defer r.deinit();
            try t.expect("inv", try dup(r.inverse), try c.get("inv"));
            try t.expect("idx_sorted", try dup(r.sorted), try c.get("idx_sorted"));
            try t.expect("xr", try h32Rows(s, x, xd), try c.get("xr"));
            try t.expect("xs", try takeRows(s, try c.get("xr"), r.lhs), try c.get("xs"));
            const sorted = try c.get("idx_sorted");
            try t.expect("h", try gatherQmmSorted(s, try c.get("xs"), lb.bank.gate, lb.bank.up, sorted, lb.bits_gu, 0, arm), try c.get("h"));
            try t.expect("hr", try h32Rows(s, try c.get("h"), xd), try c.get("hr"));
            try t.expect("y", try gatherQmmSorted(s, try c.get("hr"), lb.bank.down, null, sorted, lb.bits_dn, 0, arm), try c.get("y"));
            try t.expect("out", try weightedUnsort(s, try c.get("y"), try c.get("inv"), wts, tokens, k), out);
        }
        const limit: f32 = if (case.get("limit")) |v| switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => return error.FixtureBadLimit,
        } else 0;
        try t.expect("moe", try moeOn(s, x, lb.bank, inds, scores, limit, arm), out);
    } else if (is(u8, kind, "tl_fused")) {
        // The gate/up kernel over the banks' first `cols` inputs: a K with a partial last block.
        const words: c_int = @intCast(@divExact(case.get("cols").?.integer * lb.bits_gu, 32));
        const g = try leadingWords(lb.bank.gate, words, s);
        defer free(g.packed_w);
        const u = try leadingWords(lb.bank.up, words, s);
        defer free(u.packed_w);
        try t.expect("out", try gatherQmv(s, x, g, u, try c.get("idx"), lb.bits_gu, 0), out);
    } else return error.FixtureUnknownKind;
    std.debug.print("[jangtq2-fixture] {s} ({s}): {d}/{d} bit-identical\n", .{ name, kind, t.checks - t.fails, t.checks });
    return t.fails;
}

test "jangtq2 kernels and moe match vMLX bit for bit on the JANGH4 fixtures (JANGTQ2_FIXTURES)" {
    const dir = std.mem.span(std.c.getenv("JANGTQ2_FIXTURES") orelse return error.SkipZigTest);
    try requireMetal();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const manifest = try readJson(a, io, try std.fmt.allocPrint(a, "{s}/manifest.json", .{dir}));
    const root = manifest.value.object;
    // `prefill` names the arm the references came from (tests/dump_jangtq2_fixtures.py --steel).
    const arm: Prefill = if (root.get("prefill")) |v| (if (std.mem.eql(u8, v.string, "steel")) .steel else .nax) else .nax;
    if (arm == .nax and !@import("transformer.zig").naxAvailable()) {
        std.debug.print("[jangtq2-fixture] {s} holds NAX references and this GPU has no NAX: skipped\n", .{dir});
        return error.SkipZigTest;
    }
    const bundle = root.get("bundle").?.string;
    var cases: usize = 0;
    var failed_cases: usize = 0;
    var layer: ?i64 = null;
    var lb: ?LayerBank = null;
    defer if (lb) |b| b.deinit();
    for (root.get("cases").?.array.items) |c| {
        const case = c.object;
        const l = case.get("layer").?.integer;
        if (layer != l) {
            if (lb) |b| b.deinit();
            lb = null;
            _ = mlx.mlx_clear_cache();
            lb = try LayerBank.load(a, io, bundle, l);
            layer = l;
        }
        cases += 1;
        if (try fixtureCase(a, dir, case, lb.?, arm, s) != 0) failed_cases += 1;
    }
    std.debug.print("[jangtq2-fixture] {s} (prefill {t}): {d}/{d} cases bit-identical\n", .{ dir, arm, cases - failed_cases, cases });
    try testing.expect(cases > 0);
    try testing.expectEqual(@as(usize, 0), failed_cases);
}
