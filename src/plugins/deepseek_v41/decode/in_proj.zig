//! k1 `in_proj` (`~/plans/dsv41-decode-kernels-spec.md`): the attention input's collapse and norm, then q_kv, weights_proj, the compressor rows and the hc mixes.
//! One or two rows are the single-launch kernel (`dec_in_proj.metal`, `FUSE`: every threadgroup collapses the stream itself);
//! 3..6 rows run `dec_in_proj_stage.metal` first, so the stream is read once instead of once a threadgroup.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const device = @import("../../../mlx/device.zig");
const mlx = @import("../../../mlx/ffi.zig");
const mxfp8_qmv = @import("../../../sdk/blocks/qmm/mxfp8_qmv.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const decode = @import("decode.zig");
const v41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const MAIN_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_in_proj.metal");
const STAGE_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_in_proj_stage.metal");
/// bf16 pairs from a word, and the mxfp8 e4m3 pair decode.
const HEADER = mxfp8_qmv.HEADER ++ kernelSource("plugins/deepseek_v41/kernels/dec_in_proj_header.metal");
const MAIN_INS = [_][*:0]const u8{ "x", "cpre", "attn_w", "eps", "hin", "fn_w", "scale", "base", "wq", "wsc", "wp", "cw" };
const MAIN_OUTS = [_][*:0]const u8{ "h", "qa", "kv", "iw", "comp", "z" };
const STAGE_INS = [_][*:0]const u8{ "x", "cpre", "w", "eps" };
const STAGE_OUTS = [_][*:0]const u8{"h"};

/// The first Apple GPU family (M3) that runs its 640-thread groups at every row count: M1 and M2 cap a pipeline's
/// threads by its registers, and R = 6 with bf16 rows compiles for fewer.
const first_gen = 15;
/// The most rows whose stream every threadgroup re-reads and collapses: past two the re-reads cost more than the stage launch.
const fuse_rows = 2;
/// Output rows a threadgroup: q_kv's mxfp8 rows, and the bf16 rows of weights_proj and the compressor.
const rt = 4;
/// The stream's copies, and the hc mixes the layer makes (`(2 + hc) * hc`).
const hc = 4;
const n_mix = (2 + hc) * hc;

pub const Kernels = struct {
    main: mlx.mlx_fast_metal_kernel,
    stage: mlx.mlx_fast_metal_kernel,

    pub fn build() !Kernels {
        const main = try v41.buildKernel("dsv41_dec_in_proj", &MAIN_INS, &MAIN_OUTS, MAIN_SOURCE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(main);
        return .{ .main = main, .stage = try v41.buildKernel("dsv41_dec_in_proj_stage", &STAGE_INS, &STAGE_OUTS, STAGE_SOURCE, HEADER) };
    }

    pub fn deinit(k: Kernels) void {
        _ = mlx.mlx_fast_metal_kernel_free(k.main);
        _ = mlx.mlx_fast_metal_kernel_free(k.stage);
    }
};

/// What a launch computes: the rows, the hidden width, and which of the layer's projections ride along.
pub const Plan = struct {
    rows: usize,
    dim: usize,
    /// q_kv's rows (wq_a then wkv) and how many are wq_a's.
    nq: usize,
    qa: usize,
    /// weights_proj's rows, the compressor's rows (0: none), and whether the compressor pools in f32 (ratio 2).
    nw: usize = 0,
    nc: usize = 0,
    comp_f32: bool = false,
    /// Every threadgroup collapses the stream itself (up to `fuse_rows` rows); else `dec_in_proj_stage.metal` makes h first.
    fuse: bool = true,

    fn compDtype(p: Plan) mlx.mlx_dtype {
        return if (p.comp_f32) .float32 else .bfloat16;
    }

    fn threads(p: Plan) usize {
        return p.dim / 8;
    }

    /// q_kv, weights_proj and compressor threadgroups, then one per hc mix.
    fn groups(p: Plan) usize {
        return (p.nq + p.nw + p.nc) / rt + n_mix;
    }
};

/// Every array a launch reads (`wp`, `cw`: null where the layer has no such rows).
pub const Args = struct { x: A, cpre: A, attn_w: A, eps: A, fn_w: A, scale: A, base: A, wq: A, wsc: A, wp: ?A = null, cw: ?A = null };

pub const Out = struct {
    /// rms(collapse(x, cpre), attn_norm) `[R, D]`.
    h: A,
    /// q_kv's output split at wq_a's rows: `[R, 1280]` and `[R, 512]`.
    qa: A,
    kv: A,
    /// weights_proj `[R, 32]` bf16 and the compressor `[R, 2 * 512]` f32 (ratio 2) or `[R, 512]` bf16 (ratio 1), where the layer has them.
    iw: ?A,
    comp: ?A,
    /// This stage's mixes `[R, 24]` f32, the input of the hc split.
    z: A,
};

fn dims(a: A) []const c_int {
    return mlx.getShape(a);
}

fn is(a: A, dt: mlx.mlx_dtype, shape: []const usize) bool {
    if (mlx.mlx_array_dtype(a) != dt) return false;
    const sh = dims(a);
    if (sh.len != shape.len) return false;
    for (sh, shape) |s, w| if (s < 0 or @as(usize, @intCast(s)) != w) return false;
    return true;
}

/// A dense bf16 `[rows, D]` linear's rows, or null.
fn bf16Rows(l: *const v41.Lin, d: usize) ?usize {
    if (l.s.ctx != null or mlx.mlx_array_dtype(l.w) != .bfloat16) return null;
    const sh = dims(l.w);
    if (sh.len != 2 or sh[1] != d or sh[0] <= 0 or @mod(sh[0], rt) != 0) return null;
    return @intCast(sh[0]);
}

/// The launch `ly` takes for the stream `x` `[R, hc, D]`, or null where the kernel does not serve it: another dtype,
/// width or row count, a q_kv, weights_proj or compressor in another format, a layer without q_kv (the DSpark stages).
pub fn plan(m: *const v41.Dsv41Model, ly: *const v41.Layer, x: A) ?Plan {
    if (device.deviceGeneration().gen < first_gen) return null;
    const c = &m.cfg;
    const sh = dims(x);
    if (sh.len != 3 or mlx.mlx_array_dtype(x) != .bfloat16 or sh[0] < 1 or sh[0] > decode.max_rows) return null;
    const d: usize = c.hidden_size;
    if (sh[1] != hc or sh[2] != d or c.dsv4_hc_mult != hc or d % 256 != 0 or d > 8192) return null;
    const q = ly.q_kv orelse return null;
    if (q.qp.mode != .mxfp8 or q.qp.bits != 8 or q.qp.group_size != 32 or q.b.ctx != null) return null;
    const nq: usize = c.dsv4_q_lora_rank + c.head_dim;
    if (nq % rt != 0 or !is(q.w, .uint32, &.{ nq, d / 4 }) or !is(q.s, .uint8, &.{ nq, d / 32 })) return null;
    const f = ly.hc_attn;
    if (!is(f.fn_w, .float32, &.{ n_mix, hc * d }) or !is(f.scale, .float32, &.{n_mix}) or !is(f.base, .float32, &.{n_mix}) or !is(ly.attn_norm, .bfloat16, &.{d})) return null;
    var p: Plan = .{ .rows = @intCast(sh[0]), .dim = d, .nq = nq, .qa = c.dsv4_q_lora_rank, .fuse = sh[0] <= fuse_rows };
    if (ly.idx) |*iw| p.nw = bf16Rows(&iw.weights_proj, d) orelse return null;
    if (ly.comp) |*cw| {
        p.nc = bf16Rows(&cw.wkv, d) orelse return null;
        p.comp_f32 = ly.ratio > 1;
    }
    return p;
}

/// `plan`'s arrays from the layer.
pub fn argsOf(ly: *const v41.Layer, x: A, cpre: A, eps: A) Args {
    const q = ly.q_kv.?;
    return .{
        .x = x,
        .cpre = cpre,
        .attn_w = ly.attn_norm,
        .eps = eps,
        .fn_w = ly.hc_attn.fn_w,
        .scale = ly.hc_attn.scale,
        .base = ly.hc_attn.base,
        .wq = q.w,
        .wsc = q.s,
        .wp = if (ly.idx) |iw| iw.weights_proj.w else null,
        .cw = if (ly.comp) |cw| cw.wkv.w else null,
    };
}

/// The attention input of `ly` for `x` `[R, hc, D]` and the previous mix `cpre`, or null (see `plan`).
pub fn run(sc: *v41.Scope, k: *const Kernels, m: *const v41.Dsv41Model, ly: *const v41.Layer, x: A, cpre: A) !?Out {
    const p = plan(m, ly, x) orelse return null;
    return try launch(sc, k, p, argsOf(ly, x, cpre, try v41.hcEps(sc, m)));
}

fn keep(sc: *v41.Scope, a: A) !A {
    sc.items.append(sc.a, a) catch |e| {
        _ = mlx.mlx_array_free(a);
        return e;
    };
    return a;
}

fn output(sc: *v41.Scope, ov: mlx.mlx_vector_array, i: usize) !A {
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&o, ov, i)) catch |e| {
        _ = mlx.mlx_array_free(o);
        return e;
    };
    return keep(sc, o);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The main kernel's dispatch: its grid, its six outputs (one that a plan lacks is one element) and its template args.
fn launchGeometry(p: Plan) kc.metal.Geometry {
    var g: kc.metal.Geometry = .{ .grid = .{ ci(p.groups() * p.threads()), 1, 1 }, .tg = .{ ci(p.threads()), 1, 1 } };
    const r = ci(p.rows);
    if (p.fuse) g.out(&.{ r, ci(p.dim) }, .bfloat16) else g.out(&.{1}, .bfloat16);
    g.out(&.{ r, ci(p.qa) }, .bfloat16);
    g.out(&.{ r, ci(p.nq - p.qa) }, .bfloat16);
    g.out(&.{ r, @max(ci(p.nw), 1) }, .bfloat16);
    g.out(&.{ r, @max(ci(p.nc), 1) }, p.compDtype());
    g.out(&.{ r, n_mix }, .float32);
    inline for (.{ .{ "R", p.rows }, .{ "RT", rt }, .{ "NQ", p.nq }, .{ "QA", p.qa }, .{ "NW", p.nw }, .{ "NC", p.nc }, .{ "D", p.dim } }) |t| g.int(t[0], t[1]);
    g.flag("FUSE", p.fuse);
    g.dtype("CT", p.compDtype());
    // The outputs the plan does not make are never written.
    g.unread = @as(u8, if (p.fuse) 0 else 1) | @as(u8, if (p.nw == 0) 8 else 0) | @as(u8, if (p.nc == 0) 16 else 0);
    return g;
}

/// The stage kernel's dispatch: a threadgroup a row.
fn stageGeometry(p: Plan) kc.metal.Geometry {
    var g: kc.metal.Geometry = .{ .grid = .{ ci(p.rows * p.threads()), 1, 1 }, .tg = .{ ci(p.threads()), 1, 1 } };
    g.out(&.{ ci(p.rows), ci(p.dim) }, .bfloat16);
    g.int("D", p.dim);
    return g;
}

fn configure(g: kc.metal.Geometry) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (g.outs[0..g.n_outs]) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &o.shape, o.ndim, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.grid[0], g.grid[1], g.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, g.tg[0], g.tg[1], g.tg[2]));
    for (g.targs[0..g.n_targs]) |t| try mlx.check(switch (t.v) {
        .int => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, v),
        .flag => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, t.name, v),
        .dtype => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, v),
    });
    return cfg;
}

/// Runs `kernel` on `ins` at `g`; the outputs, in the kernel's order, are kept by `sc`.
fn dispatch(sc: *v41.Scope, kernel: mlx.mlx_fast_metal_kernel, g: kc.metal.Geometry, ins: []const A, outs: []A) !void {
    const cfg = try configure(g);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, sc.s));
    for (outs, 0..) |*o, i| o.* = try output(sc, ov, i);
}

/// The stage kernel: the normed collapse `[R, D]` of the stream, for the plans that do not collapse it in the main kernel.
fn stage(sc: *v41.Scope, k: *const Kernels, p: Plan, a: Args) !A {
    var o: [1]A = undefined;
    try dispatch(sc, k.stage, stageGeometry(p), &.{ a.x, a.cpre, a.attn_w, a.eps }, &o);
    return o[0];
}

/// The main kernel: `hin` the staged normed collapse (null where the plan collapses the stream itself).
fn project(sc: *v41.Scope, k: *const Kernels, p: Plan, a: Args, hin: ?A) !Out {
    std.debug.assert((hin == null) == p.fuse);
    var o: [6]A = undefined;
    // An input the launch does not read is the stream (bf16, like the rows it stands in for).
    try dispatch(sc, k.main, launchGeometry(p), &.{ a.x, a.cpre, a.attn_w, a.eps, hin orelse a.x, a.fn_w, a.scale, a.base, a.wq, a.wsc, a.wp orelse a.x, a.cw orelse a.x }, &o);
    return .{ .h = hin orelse o[0], .qa = o[1], .kv = o[2], .iw = if (p.nw > 0) o[3] else null, .comp = if (p.nc > 0) o[4] else null, .z = o[5] };
}

/// The launch itself: for `a`'s arrays, as `p` says.
pub fn launch(sc: *v41.Scope, k: *const Kernels, p: Plan, a: Args) !Out {
    return project(sc, k, p, a, if (p.fuse) null else try stage(sc, k, p, a));
}

// ── the kernel contract ───────────────────────────────────────────────────

const kit = @import("../../deepseek_v4/contract_kit.zig");
const contract = @import("../contract.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;
const Iv = contract.Iv;

const DIM = 5120;
const EPS_RMS = 1e-20;
const EPS_HC = 1e-6;
/// fp32's unit roundoff.
const u = 0x1p-24;
/// The layer cut to a few q_kv rows (24 of wq_a, 8 of wkv): rows are independent threadgroups, while the reduction
/// (the hidden width) and the stream stay the release's.
const NQ = 32;
const QA = 24;

const main_l: metal.Launch = .{
    .name = "dsv41_dec_in_proj",
    .inputs = &MAIN_INS,
    .outputs = &MAIN_OUTS,
    .source = MAIN_SOURCE,
    .header = HEADER,
    .tg_bytes = 4 * (rt + 2) * decode.max_rows * (DIM / 8 / 32),
    // M1 and M2 compile 6 rows with bf16 rows for 576 threads: the wrapper declines them (`first_gen`).
    .m1 = 576,
    .m2 = 576,
    .min_gen = first_gen,
};
const stage_l: metal.Launch = .{
    .name = "dsv41_dec_in_proj_stage",
    .inputs = &STAGE_INS,
    .outputs = &STAGE_OUTS,
    .source = STAGE_SOURCE,
    .header = HEADER,
    .tg_bytes = 4 * (DIM / 8 / 32),
    .m1 = 1024,
    .m2 = 1024,
    .min_gen = first_gen,
};

fn bf(v: f64) f64 {
    return spec.roundTo(.bf16, v);
}

/// An e4m3 code as its value (the NaN codes are never filled).
fn e4m3(c: u8) f64 {
    const s: f64 = if (c & 0x80 != 0) -1 else 1;
    const e: i32 = (c >> 3) & 15;
    const m: f64 = @floatFromInt(c & 7);
    if (e == 0) return s * m / 8 * 0x1p-6;
    return s * (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// The code of a small integer (|v| <= 8), which e4m3 holds exactly.
fn e4m3Int(v: f64) u8 {
    if (v == 0) return 0;
    const a = @abs(v);
    const e: i32 = @intFromFloat(@floor(@log2(a)));
    const m: u8 = @intFromFloat((a / std.math.pow(f64, 2, @floatFromInt(e)) - 1) * 8);
    return (if (v < 0) @as(u8, 0x80) else 0) | @as(u8, @intCast(e + 7)) << 3 | m;
}

/// Four codes a word, little-endian, as the f64 values an input holds.
fn packCodes(codes: []const u8, words: []f64) void {
    for (words, 0..) |*w, i| w.* = @floatFromInt(@as(u32, codes[4 * i]) | @as(u32, codes[4 * i + 1]) << 8 | @as(u32, codes[4 * i + 2]) << 16 | @as(u32, codes[4 * i + 3]) << 24);
}

/// Row `n` of mxfp8 weights (packed words, e8m0 scales) as values, `out.len` wide.
fn mxRow(words: []const f64, scales: []const f64, n: usize, out: []f64) void {
    const d = out.len;
    for (out, 0..) |*o, i| {
        const word: u32 = @intFromFloat(words[n * (d / 4) + i / 4]);
        const code: u8 = @truncate(word >> @intCast(8 * (i % 4)));
        const e: i32 = @intFromFloat(scales[n * (d / 32) + i / 32]);
        o.* = e4m3(code) * std.math.pow(f64, 2, @floatFromInt(e - 127));
    }
}

/// One row's normed collapse `T(w * T(T(h) * inv))` in float64 with the sums as `sum` asks: `h` the result, `mag` its size.
fn stageRow(sum: Sum, a: std.mem.Allocator, x: []const f64, cpre: []const f64, w: []const f64, eps: f64, h: []f64, mag: []f64) !void {
    const d = w.len;
    const t = try a.alloc(f64, 2 * d);
    defer a.free(t);
    for (0..d) |i| {
        var terms: [hc]f64 = undefined;
        for (&terms, 0..) |*o, j| o.* = cpre[j] * x[j * d + i];
        t[i] = bf(sum.of(&terms));
        t[d + i] = t[i] * t[i];
    }
    const inv = 1 / @sqrt(sum.of(t[d..]) / @as(f64, @floatFromInt(d)) + eps);
    for (0..d) |i| {
        h[i] = bf(w[i] * bf(t[i] * inv));
        mag[i] = @abs(h[i]);
    }
}

/// The kernel's normed rows (bf16 from the start of its output) inside the intervals of their roundings (`point`:
/// float64 classes); inputs 0..3 are the stream, `cpre`, the norm weights and the eps.
fn normedOk(k: anytype, point: bool) !?[]const u8 {
    const d = k.shape.k;
    const a = k.scratch();
    const hm = try a.alloc(f64, 2 * d);
    defer a.free(hm);
    const iv = try a.alloc(Iv, d);
    defer a.free(iv);
    const eps = spec.roundTo(.f32, k.in(3)[0]);
    for (0..k.shape.rows) |r| {
        const x = k.in(0)[r * hc * d ..][0 .. hc * d];
        const cp = k.in(1)[r * hc ..][0..hc];
        for (0..d) |i| {
            hm[i] = 0;
            hm[d + i] = 0;
            for (0..hc) |j| {
                hm[i] += cp[j] * x[j * d + i];
                hm[d + i] += @abs(cp[j] * x[j * d + i]);
            }
        }
        contract.collapseNorm(hm[0..d], hm[d..], k.in(2), eps, point, iv);
        for (iv, 0..) |q, i| {
            const g = spec.read(.bf16, k.rig.outputBytes(), r * d + i);
            const ok = if (point) spec.classOf(g) == spec.classOf(q.lo) else g >= q.lo and g <= q.hi;
            if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} h {d} is {e} outside [{e}, {e}]", .{ r, i, g, q.lo, q.hi }) catch "h";
        }
    }
    return null;
}

/// The stream is +-32 everywhere (every sum of its squares is exact, the rms norm exactly 1 at eps 0); with `plus_first`
/// its first copy is +32, so the collapse under `cpre` = (1, 0, 0, 0) is 32 and the normed collapse is the norm weights.
fn stressStream(rng: std.Random, x: []f64, rows: usize, d: usize, plus_first: bool) void {
    const n = hc * d;
    for (x[0..n], 0..) |*v, i| v.* = if (plus_first and i < d) 32 else if (rng.boolean()) 32 else -32;
    for (1..rows) |r| @memcpy(x[r * n ..][0..n], x[0..n]);
}

/// The mixes' weights for such a stream (`HcStage`'s): the first half's products are 32 * (2^10 + a digit), the second
/// half's -32 * 2^10, so partial sums reach 2^28 and the total is a small multiple of 32.
fn stressMixes(rng: std.Random, fnw: []f64, x: []const f64) void {
    const n = x.len;
    const h = n / 2;
    for (0..n_mix) |m| for (0..h) |i| {
        const b: f64 = 0x1p10 + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 0, 8)));
        fnw[m * n + i] = b * x[i] / 32;
        fnw[m * n + h + i] = -0x1p10 * x[h + i] / 32;
    };
}

fn depthMixes(_: Shape) usize {
    return hc * DIM + 8;
}

fn depthNorm(s: Shape) usize {
    return s.k + 8;
}

/// The main kernel with `fuse` (every threadgroup collapses the stream; else `hin` holds the normed collapse), `nw`
/// weights_proj rows and `nc` compressor rows (f32 when `cf32`).
fn Main(comptime fuse: bool, comptime nw: usize, comptime nc: usize, comptime cf32: bool) type {
    return struct {
        pub const Params = struct {};
        const comp_elem: spec.Elem = if (cf32) .f32 else .bf16;

        fn planOf(s: Shape) Plan {
            return .{ .rows = s.rows, .dim = s.k, .nq = NQ, .qa = QA, .nw = nw, .nc = nc, .comp_f32 = cf32, .fuse = fuse };
        }

        pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
            if (s.rows < 1 or s.rows > decode.max_rows or s.k != DIM) return null;
            return launchGeometry(planOf(s));
        }

        /// The outputs in the kernel's order: h (this arm makes it), q_kv's two halves and weights_proj in bf16, then the
        /// compressor, then the mixes in f32.
        pub fn bf16s(s: Shape) usize {
            return (if (fuse) s.rows * s.k else 0) + s.rows * (NQ + nw);
        }

        pub fn comps(s: Shape) usize {
            return s.rows * nc;
        }

        pub fn mixes(s: Shape) usize {
            return s.rows * n_mix;
        }

        pub fn count(_: Params, s: Shape) usize {
            return bf16s(s) + comps(s) + mixes(s);
        }

        /// Dot `i` of row `r` (a q_kv, weights_proj, then compressor row) in the outputs, as an element index.
        fn dotAt(s: Shape, r: usize, i: usize) usize {
            const q = if (fuse) s.rows * s.k else 0;
            if (i < QA) return q + r * QA + i;
            if (i < NQ) return q + s.rows * QA + r * (NQ - QA) + (i - QA);
            if (i < NQ + nw) return q + s.rows * NQ + r * nw + (i - NQ);
            return bf16s(s) + r * nc + (i - NQ - nw);
        }

        /// Weight row `i` of the dots as values.
        fn weightRow(k: anytype, i: usize, w: []f64) void {
            const d = k.shape.k;
            if (i < NQ) mxRow(k.in(8), k.in(9), i, w) else if (i < NQ + nw) @memcpy(w, k.in(10)[(i - NQ) * d ..][0..d]) else @memcpy(w, k.in(11)[(i - NQ - nw) * d ..][0..d]);
        }

        pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
            const rows = k.shape.rows;
            const d = k.shape.k;
            const n = hc * d;
            const x = try k.add(.bfloat16, &.{ rows, hc, d });
            const cpre = try k.add(.float32, &.{ rows, hc });
            const aw = try k.add(.bfloat16, &.{d});
            const eps = try k.add(.float32, &.{2});
            const hin = try k.add(.bfloat16, if (fuse) &[_]usize{16} else &[_]usize{ rows, d });
            const fnw = try k.add(.float32, &.{ n_mix, n });
            const scale = try k.add(.float32, &.{n_mix});
            const base = try k.add(.float32, &.{n_mix});
            const wq = try k.add(.uint32, &.{ NQ, d / 4 });
            const wsc = try k.add(.uint8, &.{ NQ, d / 32 });
            const wp = try k.add(.bfloat16, if (nw > 0) &[_]usize{ nw, d } else &[_]usize{16});
            const cw = try k.add(.bfloat16, if (nc > 0) &[_]usize{ nc, d } else &[_]usize{16});
            kit.uniform(rng, x, -3, 3);
            kit.uniform(rng, cpre, 0.05, 0.45);
            kit.uniform(rng, aw, 0.5, 1.5);
            eps[0] = EPS_RMS;
            eps[1] = EPS_HC;
            if (!fuse) kit.uniform(rng, hin, -2, 2);
            kit.uniform(rng, fnw, -0.02, 0.02);
            kit.uniform(rng, scale, 0.5, 1.5);
            kit.uniform(rng, base, -0.5, 0.5);
            const codes = try k.scratch().alloc(u8, NQ * d);
            defer k.scratch().free(codes);
            for (codes) |*c| c.* = (if (rng.boolean()) @as(u8, 0x80) else 0) | rng.intRangeAtMost(u8, 1, 10) << 3 | rng.intRangeAtMost(u8, 0, 7);
            packCodes(codes, wq);
            for (wsc) |*v| v.* = @floatFromInt(rng.intRangeAtMost(u8, 118, 121));
            kit.uniform(rng, wp, -0.05, 0.05);
            kit.uniform(rng, cw, -0.05, 0.05);
            switch (f) {
                .random => {},
                .nonfinite => {
                    kit.plantRows(x, rows, n);
                    if (!fuse) kit.plantRows(hin, rows, d);
                },
                .stress => {
                    // Every dot is a mirrored-integer row of `stressDot` against h, which is the pattern: the norm
                    // weights where the kernel collapses (the collapse is 32 over an rms of 32), else the input itself.
                    const a = k.scratch();
                    const pat = try a.alloc(f64, d);
                    defer a.free(pat);
                    const wr = try a.alloc(f64, (NQ + nw + nc) * d);
                    defer a.free(wr);
                    try contract.stressDot(rng, pat, wr, NQ + nw + nc, d, 64, 7, a);
                    eps[0] = 0;
                    @memset(scale, 1);
                    @memset(base, 0);
                    stressStream(rng, x, rows, d, fuse);
                    stressMixes(rng, fnw, x[0..n]);
                    for (0..rows) |r| for (0..hc) |j| {
                        cpre[r * hc + j] = if (j == 0) 1 else 0;
                    };
                    if (fuse) @memcpy(aw, pat) else for (0..rows) |r| @memcpy(hin[r * d ..][0..d], pat);
                    for (codes, wr[0 .. NQ * d]) |*c, v| c.* = e4m3Int(v);
                    packCodes(codes, wq);
                    @memset(wsc, 127);
                    if (nw > 0) @memcpy(wp, wr[NQ * d ..][0 .. nw * d]);
                    if (nc > 0) @memcpy(cw, wr[(NQ + nw) * d ..][0 .. nc * d]);
                },
            }
        }

        /// The outputs in the kernel's order in float64, every sum as `sum` asks (gate 4).
        pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
            const rows = k.shape.rows;
            const d = k.shape.k;
            const n = hc * d;
            const a = k.scratch();
            const eps = spec.roundTo(.f32, k.in(3)[0]);
            const h = try a.alloc(f64, rows * d);
            defer a.free(h);
            if (fuse) {
                for (0..rows) |r| try stageRow(sum, a, k.in(0)[r * n ..][0..n], k.in(1)[r * hc ..][0..hc], k.in(2), eps, out[r * d ..][0..d], mag[r * d ..][0..d]);
                @memcpy(h, out[0 .. rows * d]);
            } else @memcpy(h, k.in(4)[0 .. rows * d]);
            const t = try a.alloc(f64, n);
            defer a.free(t);
            const w = try a.alloc(f64, d);
            defer a.free(w);
            for (0..NQ + nw + nc) |i| {
                weightRow(k, i, w);
                for (0..rows) |r| {
                    for (t[0..d], h[r * d ..][0..d], w) |*o, hv, wv| o.* = hv * wv;
                    const at = dotAt(k.shape, r, i);
                    out[at] = sum.of(t[0..d]);
                    mag[at] = 0;
                    for (t[0..d]) |v| mag[at] += @abs(v);
                }
            }
            for (0..rows) |r| {
                const xr = k.in(0)[r * n ..][0..n];
                for (t, xr) |*o, v| o.* = v * v;
                const rs = 1 / @sqrt(sum.of(t) / @as(f64, @floatFromInt(n)) + eps);
                for (0..n_mix) |m| {
                    for (t, xr, k.in(5)[m * n ..][0..n]) |*o, v, f| o.* = v * f;
                    const at = bf16s(k.shape) + comps(k.shape) + r * n_mix + m;
                    out[at] = sum.of(t) * rs * k.in(6)[m] + k.in(7)[m];
                    mag[at] = @abs(out[at]);
                }
            }
        }

        /// The mixes within their allowances, the normed collapse within the intervals of its roundings, and every dot
        /// within its sums' allowance on the h the kernel used.
        pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
            const rows = k.shape.rows;
            const d = k.shape.k;
            const n = hc * d;
            const point = at == .nonfinite;
            const a = k.scratch();
            const bytes = k.rig.outputBytes();
            const comp_at = 2 * bf16s(k.shape);
            const z_at = comp_at + comps(k.shape) * @as(usize, if (cf32) 4 else 2);
            const eps = spec.roundTo(.f32, k.in(3)[0]);
            var z: [n_mix]f64 = undefined;
            var al: [n_mix]f64 = undefined;
            for (0..rows) |r| {
                contract.mixRow(k.in(0)[r * n ..][0..n], k.in(5), k.in(6), k.in(7), eps, &z, &al);
                for (z, al, 0..) |want, allow, m| {
                    const g = spec.read(.f32, bytes[z_at..], r * n_mix + m);
                    const ok = if (point) spec.classOf(g) == spec.classOf(want) else @abs(g - want) <= allow;
                    if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} mix {d} is {e} where float64 gives {e}", .{ r, m, g, want }) catch "z";
                }
            }
            if (fuse) if (try normedOk(k, point)) |why| return why;
            const h = try a.alloc(f64, rows * d);
            defer a.free(h);
            for (h, 0..) |*o, i| o.* = if (fuse) spec.read(.bf16, bytes, i) else k.in(4)[i];
            const w = try a.alloc(f64, d);
            defer a.free(w);
            const depth: f64 = @floatFromInt(8 + 5 + d / 256);
            for (0..NQ + nw + nc) |i| {
                weightRow(k, i, w);
                const in_comp = i >= NQ + nw;
                for (0..rows) |r| {
                    var dot: f64 = 0;
                    var mag: f64 = 0;
                    for (h[r * d ..][0..d], w) |hv, wv| {
                        dot += hv * wv;
                        mag += @abs(hv * wv);
                    }
                    const at_i = dotAt(k.shape, r, i);
                    const elem: spec.Elem = if (in_comp) comp_elem else .bf16;
                    const g = if (in_comp) spec.read(elem, bytes[comp_at..], at_i - bf16s(k.shape)) else spec.read(.bf16, bytes, at_i);
                    const work = spec.gamma(depth, u) * mag;
                    const unit: f64 = if (elem == .bf16) 0x1p-8 else u;
                    const tiny: f64 = if (elem == .bf16) 0x1p-133 else 0x1p-149;
                    const ok = if (point) spec.classOf(g) == spec.classOf(dot) else @abs(g - dot) <= work + unit * (@abs(dot) + work) + tiny;
                    if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} dot {d} is {e} where float64 gives {e}", .{ r, i, g, dot }) catch "dot";
                }
            }
            return null;
        }
    };
}

/// The stage kernel: the normed collapse of each row, a threadgroup a row.
const Stage = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows < 1 or s.rows > decode.max_rows or s.k != DIM) return null;
        return stageGeometry(.{ .rows = s.rows, .dim = s.k, .nq = 0, .qa = 0 });
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.k;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const rows = k.shape.rows;
        const d = k.shape.k;
        const x = try k.add(.bfloat16, &.{ rows, hc, d });
        const cpre = try k.add(.float32, &.{ rows, hc });
        const w = try k.add(.bfloat16, &.{d});
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, x, -3, 3);
        kit.uniform(rng, cpre, 0.05, 0.45);
        kit.uniform(rng, w, 0.5, 1.5);
        eps[0] = EPS_RMS;
        eps[1] = EPS_HC;
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, rows, hc * d),
            .stress => {
                const a = k.scratch();
                const pat = try a.alloc(f64, d);
                defer a.free(pat);
                const scrap = try a.alloc(f64, d);
                defer a.free(scrap);
                try contract.stressDot(rng, pat, scrap, 1, d, 64, 7, a);
                eps[0] = 0;
                stressStream(rng, x, rows, d, true);
                for (0..rows) |r| for (0..hc) |j| {
                    cpre[r * hc + j] = if (j == 0) 1 else 0;
                };
                @memcpy(w, pat);
            },
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const d = k.shape.k;
        const eps = spec.roundTo(.f32, k.in(3)[0]);
        for (0..k.shape.rows) |r| try stageRow(sum, k.scratch(), k.in(0)[r * hc * d ..][0 .. hc * d], k.in(1)[r * hc ..][0..hc], k.in(2), eps, out[r * d ..][0..d], mag[r * d ..][0..d]);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        return normedOk(k, at == .nonfinite);
    }
};

const nan_rule: spec.Nan = .{ .stated = "classes as float64 gives them; the normed rows within the intervals of their roundings, the dots and mixes within their sums' allowance" };
const fused_shapes: []const Shape = &.{ .{ .rows = 1, .n = NQ, .k = DIM }, .{ .rows = fuse_rows, .n = NQ, .k = DIM } };
const staged_shapes: []const Shape = &.{ .{ .rows = fuse_rows + 1, .n = NQ, .k = DIM }, .{ .rows = decode.max_rows, .n = NQ, .k = DIM } };

fn addMain(set: *kc.Set, comptime fuse: bool, comptime nw: usize, comptime nc: usize, comptime cf32: bool, comptime name: []const u8) !void {
    const T = Main(fuse, nw, nc, cf32);
    try kit.add(set, T, &main_l, .{}, .{
        .name = name,
        .accum = .f32,
        .depth = depthMixes,
        .ops = 4,
        .mids = 2,
        .out = .bf16,
        .outs = &.{ .{ .elem = .bf16, .count = T.bf16s }, .{ .elem = T.comp_elem, .count = T.comps }, .{ .elem = .f32, .count = T.mixes } },
        .nan = nan_rule,
        .served = if (fuse) fused_shapes else staged_shapes,
    });
}

pub fn register(set: *kc.Set) !void {
    try addMain(set, true, 0, 0, false, "dsv41 dec in_proj, 1..2 rows (q_kv, hc mixes)");
    try addMain(set, true, 8, 16, true, "dsv41 dec in_proj, 1..2 rows (+ weights_proj, f32 compressor)");
    try addMain(set, true, 8, 16, false, "dsv41 dec in_proj, 1..2 rows (+ weights_proj, bf16 compressor)");
    try addMain(set, false, 0, 0, false, "dsv41 dec in_proj, 3..6 rows (q_kv, hc mixes)");
    try addMain(set, false, 8, 16, true, "dsv41 dec in_proj, 3..6 rows (+ weights_proj, f32 compressor)");
    try addMain(set, false, 8, 16, false, "dsv41 dec in_proj, 3..6 rows (+ weights_proj, bf16 compressor)");
    try kit.add(set, Stage, &stage_l, .{}, .{
        .name = "dsv41 dec in_proj_stage (normed collapse)",
        .accum = .f32,
        .depth = depthNorm,
        .ops = 4,
        .mids = 2,
        .out = .bf16,
        .nan = nan_rule,
        .served = &.{ .{ .rows = fuse_rows + 1, .n = DIM, .k = DIM }, .{ .rows = decode.max_rows, .n = DIM, .k = DIM } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "a sum of squares cannot cancel" },
    });
}

test "dsv41 dec in_proj: the declared launches are the shipped wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const kn = try Kernels.build();
    defer kn.deinit();
    var sc = v41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    inline for (.{ .{ true, 0, 0, false, 1 }, .{ true, 8, 16, true, 2 }, .{ true, 8, 16, false, 1 }, .{ false, 0, 0, false, 3 }, .{ false, 8, 16, true, 6 }, .{ false, 8, 16, false, 4 } }) |v| {
        const T = Main(v[0], v[1], v[2], v[3]);
        const shape: Shape = .{ .rows = v[4], .n = NQ, .k = DIM };
        var c = kit.Case(T).init(gpa, &main_l, .{});
        defer c.deinit();
        try testing.expect(try c.prepare(shape, .random, 7));
        try c.run();
        const p = T.planOf(shape);
        const a = Args{ .x = c.args[0], .cpre = c.args[1], .attn_w = c.args[2], .eps = c.args[3], .fn_w = c.args[5], .scale = c.args[6], .base = c.args[7], .wq = c.args[8], .wsc = c.args[9], .wp = if (v[1] > 0) c.args[10] else null, .cw = if (v[2] > 0) c.args[11] else null };
        const out = try project(&sc, &kn, p, a, if (v[0]) null else c.args[4]);
        var theirs: [6]A = undefined;
        var n: usize = 0;
        for ([_]?A{ if (v[0]) out.h else null, out.qa, out.kv, out.iw, out.comp, out.z }) |o| if (o) |arr| {
            theirs[n] = arr;
            n += 1;
        };
        try testing.expectEqualSlices(u8, c.rig.outputBytes(), try c.rig.bytesOf(theirs[0..n]));
        if (!v[0]) {
            try c.rig.launch(&stage_l, stageGeometry(p), c.args[0..4]);
            try testing.expectEqualSlices(u8, c.rig.outputBytes(), try c.rig.bytesOf(&.{try stage(&sc, &kn, p, a)}));
        }
    }
}

test "dsv41 dec in_proj: the registered kernels pass the contract's gates 1-4" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var set = kc.Set.init(gpa);
    defer set.deinit();
    try register(&set);
    var t: kit.gates.Tally = .{};
    try kit.gates.runAll(gpa, set.kernels.items, 0xc0de, &t);
    std.debug.print("[dec] in_proj contract: {d}/{d} launches pass gates 1-4 (virtual GPU {})\n", .{ t.passed, t.kernels, mlx.virtualGpu() });
    try testing.expectEqual(@as(usize, 0), t.problems());
    try testing.expectEqual(set.kernels.items.len, t.kernels);
}

// ── tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;
const fixture = decode.fixture;

fn host(a: A) ![]f32 {
    return v41.hostF32(a);
}

fn asF32(sc: *v41.Scope, a: A) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, a, .float32, sc.s));
    return keep(sc, o);
}

/// `hcStage`'s first kernel: the normed collapse and the mixes (which `hcStage` hands to the split and does not return).
fn todayStage(sc: *v41.Scope, m: *const v41.Dsv41Model, x: A, cpre: A, ly: *const v41.Layer) ![2]A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const L = dims(x)[0];
    const D = dims(x)[2];
    const tg: c_int = @intCast(m.hc_tg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ L, D }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ L, n_mix }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, (n_mix + 1) * tg, L, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", tg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ND", @divFloor(D + tg - 1, tg)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, cpre, ly.attn_norm, ly.hc_attn.fn_w, ly.hc_attn.scale, ly.hc_attn.base, try v41.hcEps(sc, m) }, 7);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.hc_stage_kernel, iv, cfg, sc.s));
    return .{ try output(sc, ov, 0), try output(sc, ov, 1) };
}

/// Float64 `h [R, N] = x [R, K] . w [N, K]^T`, as f32.
fn gemvTruth(gpa: std.mem.Allocator, x: []const f32, w: []const f32, rows: usize, n: usize, k: usize) ![]f32 {
    const out = try gpa.alloc(f32, rows * n);
    for (0..rows) |r| for (0..n) |i| {
        var s: f64 = 0;
        for (x[r * k ..][0..k], w[i * k ..][0..k]) |a, b| s += @as(f64, a) * @as(f64, b);
        out[r * n + i] = @floatCast(s);
    };
    return out;
}

fn bf16(v: f64) f64 {
    return kc.spec.roundTo(.bf16, v);
}

/// The normed collapse and the mixes in float64, with the collapse and the product rounded to bf16 where the model rounds them.
fn stageTruth(gpa: std.mem.Allocator, m: *const v41.Dsv41Model, ly: *const v41.Layer, x: []const f32, cpre: []const f32, rows: usize) ![2][]f32 {
    const d = m.cfg.hidden_size;
    const nw = try host(ly.attn_norm);
    defer testing.allocator.free(nw);
    const fnw = try host(ly.hc_attn.fn_w);
    defer testing.allocator.free(fnw);
    const sc = try host(ly.hc_attn.scale);
    defer testing.allocator.free(sc);
    const bs = try host(ly.hc_attn.base);
    defer testing.allocator.free(bs);
    const h = try gpa.alloc(f32, rows * d);
    const z = try gpa.alloc(f32, rows * n_mix);
    const hv = try gpa.alloc(f64, d);
    defer gpa.free(hv);
    const eps: f64 = m.cfg.rms_norm_eps;
    for (0..rows) |r| {
        const xr = x[r * hc * d ..][0 .. hc * d];
        var ss: f64 = 0;
        for (hv, 0..) |*o, i| {
            var s: f64 = 0;
            for (0..hc) |j| s += @as(f64, cpre[r * hc + j]) * @as(f64, xr[j * d + i]);
            o.* = bf16(s);
            ss += o.* * o.*;
        }
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + eps);
        for (hv, nw, 0..) |v, w, i| h[r * d + i] = @floatCast(bf16(@as(f64, w) * bf16(v * inv)));
        var sq: f64 = 0;
        for (xr) |v| sq += @as(f64, v) * @as(f64, v);
        const rs = 1 / @sqrt(sq / @as(f64, @floatFromInt(hc * d)) + eps);
        for (0..n_mix) |i| {
            var dd: f64 = 0;
            for (xr, fnw[i * hc * d ..][0 .. hc * d]) |a, b| dd += @as(f64, a) * @as(f64, b);
            z[r * n_mix + i] = @floatCast(dd * rs * @as(f64, sc[i]) + @as(f64, bs[i]));
        }
    }
    return .{ h, z };
}

/// q_kv's rows and wq_a's split of `[R, nq]` back to one array on the host.
fn joinRows(gpa: std.mem.Allocator, qa: []const f32, kv: []const f32, rows: usize) ![]f32 {
    const na = qa.len / rows;
    const nk = kv.len / rows;
    const out = try gpa.alloc(f32, qa.len + kv.len);
    for (0..rows) |r| {
        @memcpy(out[r * (na + nk) ..][0..na], qa[r * na ..][0..na]);
        @memcpy(out[r * (na + nk) + na ..][0..nk], kv[r * nk ..][0..nk]);
    }
    return out;
}

fn dequantHost(sc: *v41.Scope, l: *const v41.Lin) ![]f32 {
    var d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&d, l.w, l.s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, sc.s));
    return host(try keep(sc, d));
}

fn expectLayer(gpa: std.mem.Allocator, fx: *fixture.Fixture, k: *const Kernels, li: usize, rows: usize) !void {
    const m = fx.m;
    const ly = fx.layer(li);
    const d = m.cfg.hidden_size;
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const cpre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(cpre);
    var sc = v41.Scope.init(gpa, m.s);
    defer sc.deinit();
    std.debug.print("[dec] in_proj layer {d} rows {d}\n", .{ li, rows });

    const out = (try run(&sc, k, m, ly, x, cpre)) orelse return error.Declined;
    const staged = try v41.hcStage(&sc, m, x, null, cpre, &ly.hc_attn, ly.attn_norm);
    const tz = try todayStage(&sc, m, x, cpre, ly);
    const h_chain = try host(staged.h);
    defer testing.allocator.free(h_chain);
    const h_new = try host(out.h);
    defer testing.allocator.free(h_new);
    const xh = try host(x);
    defer testing.allocator.free(xh);
    const ch = try host(cpre);
    defer testing.allocator.free(ch);
    const truth = try stageTruth(gpa, m, ly, xh, ch, rows);
    defer for (truth) |t| gpa.free(t);

    // The normed collapse: bits of today's kernel but for the odd bf16 step where the sum of squares sums in another order.
    var diff: usize = 0;
    for (h_new, h_chain) |a, b| diff += @intFromBool(a != b);
    std.debug.print("[dec] h differs from today's in {d} of {d} elements\n", .{ diff, h_new.len });
    try testing.expect(diff * 500 <= h_new.len);
    try decode.expectNoWorse("h", h_new, h_chain, truth[0]);
    const zc = try host(tz[1]);
    defer testing.allocator.free(zc);
    const zn = try host(out.z);
    defer testing.allocator.free(zn);
    try decode.expectNoWorse("z", zn, zc, truth[1]);

    // The projections are judged on today's h (the dots' own input).
    const q = &ly.q_kv.?;
    const wq = try dequantHost(&sc, q);
    defer testing.allocator.free(wq);
    const y = try host(try q.apply(&sc, staged.h));
    defer testing.allocator.free(y);
    const qa = try host(out.qa);
    defer testing.allocator.free(qa);
    const kv = try host(out.kv);
    defer testing.allocator.free(kv);
    const yn = try joinRows(gpa, qa, kv, rows);
    defer gpa.free(yn);
    const yt = try gemvTruth(gpa, h_chain, wq, rows, y.len / rows, d);
    defer gpa.free(yt);
    try decode.expectNoWorse("q_kv", yn, y, yt);

    if (ly.idx) |*iw| {
        const w = try host(iw.weights_proj.w);
        defer testing.allocator.free(w);
        const want = try host(try iw.weights_proj.apply(&sc, staged.h));
        defer testing.allocator.free(want);
        const got = try host(out.iw.?);
        defer testing.allocator.free(got);
        const t = try gemvTruth(gpa, h_chain, w, rows, want.len / rows, d);
        defer gpa.free(t);
        try decode.expectNoWorse("weights_proj", got, want, t);
    } else try testing.expect(out.iw == null);
    if (ly.comp) |*cw| {
        const w = try host(cw.wkv.w);
        defer testing.allocator.free(w);
        const chain = if (ly.ratio > 1) try v41.denseF32x(&sc, m, &cw.wkv, try asF32(&sc, staged.h)) else try cw.wkv.apply(&sc, staged.h);
        const want = try host(chain);
        defer testing.allocator.free(want);
        const got = try host(out.comp.?);
        defer testing.allocator.free(got);
        const t = try gemvTruth(gpa, h_chain, w, rows, want.len / rows, d);
        defer gpa.free(t);
        try decode.expectNoWorse(if (ly.ratio > 1) "compressor (f32)" else "compressor (bf16)", got, want, t);
    } else try testing.expect(out.comp == null);
}

test "dsv41 dec in_proj: layers 20..25 and 2..4 (KV sources, index sources, plain) match today's chain at R = 1, 2, 4, 6" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const k = try Kernels.build();
    defer k.deinit();
    for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 } }) |rg| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = rg[0], .hi = rg[1], .ctx = 0 });
        defer fx.deinit();
        for (rg[0]..rg[1]) |li| for ([_]usize{ 1, 2, 4, 6 }) |rows| try expectLayer(gpa, fx, &k, li, rows);
    }
}

// ── ubench ────────────────────────────────────────────────────────────────

/// `a` in new memory (an add of zero), evaluated.
fn distinct(sc: *v41.Scope, a: A) !A {
    var zero = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 1, mlx.mlx_array_dtype(a), sc.s));
    _ = try keep(sc, zero);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, zero, sc.s));
    _ = try keep(sc, o);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// The layer's weights in `n` distinct copies, so a chain over them reads memory the system cache does not hold.
const Cycle = struct {
    n: usize,
    wq: []A,
    wsc: []A,
    fnw: []A,
    wp: []A,
    cw: []A,

    fn init(gpa: std.mem.Allocator, sc: *v41.Scope, ly: *const v41.Layer, bytes: usize, floor_bytes: usize) !Cycle {
        const n = (floor_bytes + bytes - 1) / bytes;
        const a = argsOf(ly, none, none, none);
        var c: Cycle = .{ .n = n, .wq = try gpa.alloc(A, n), .wsc = try gpa.alloc(A, n), .fnw = try gpa.alloc(A, n), .wp = try gpa.alloc(A, if (a.wp != null) n else 0), .cw = try gpa.alloc(A, if (a.cw != null) n else 0) };
        for (0..n) |i| {
            c.wq[i] = try distinct(sc, a.wq);
            c.wsc[i] = try distinct(sc, a.wsc);
            c.fnw[i] = try distinct(sc, a.fn_w);
            if (a.wp) |w| c.wp[i] = try distinct(sc, w);
            if (a.cw) |w| c.cw[i] = try distinct(sc, w);
        }
        return c;
    }

    fn deinit(c: Cycle, gpa: std.mem.Allocator) void {
        for ([_][]A{ c.wq, c.wsc, c.fnw, c.wp, c.cw }) |s| gpa.free(s);
    }

    fn args(c: Cycle, i: usize, base: Args) Args {
        var a = base;
        a.wq = c.wq[i % c.n];
        a.wsc = c.wsc[i % c.n];
        a.fn_w = c.fnw[i % c.n];
        if (c.wp.len > 0) a.wp = c.wp[i % c.n];
        if (c.cw.len > 0) a.cw = c.cw[i % c.n];
        return a;
    }
};

/// Frees what the chain before made, so each chain reuses the allocator's buffers as consecutive decode steps do.
fn restart(sc: *v41.Scope) void {
    for (sc.items.items) |x| _ = mlx.mlx_array_free(x);
    sc.items.clearRetainingCapacity();
}

/// The kernel's steps: each stage's mixes are the next one's `cpre` (the kernel reads its first R * 4), a real dependence.
const Chain = struct {
    sc: *v41.Scope,
    k: *const Kernels,
    p: Plan,
    cyc: *const Cycle,
    base: Args,
    z: A = none,

    pub fn step(c: *Chain, i: usize) !A {
        if (i == 0) restart(c.sc);
        var a = c.cyc.args(i, c.base);
        if (i > 0) a.cpre = c.z;
        c.z = (try launch(c.sc, c.k, c.p, a)).z;
        return c.z;
    }
};

/// Today's chain on the same weights: the stage kernel, q_kv, weights_proj and the compressor's projection. The next
/// step's `cpre` is the last array it made, tied to the rest of them (`mlx_depends`: no work, but all of it must run).
const OldChain = struct {
    sc: *v41.Scope,
    m: *const v41.Dsv41Model,
    ly: *const v41.Layer,
    cyc: *const Cycle,
    x: A,
    cpre0: A,
    last: A = none,

    pub fn step(c: *OldChain, i: usize) !A {
        if (i == 0) restart(c.sc);
        var ly = c.ly.*;
        const a = c.cyc.args(i, argsOf(c.ly, c.x, c.cpre0, none));
        ly.hc_attn.fn_w = a.fn_w;
        ly.q_kv.?.w = a.wq;
        ly.q_kv.?.s = a.wsc;
        if (a.wp) |w| ly.idx.?.weights_proj.w = w;
        if (a.cw) |w| ly.comp.?.wkv.w = w;
        const sc = c.sc;
        const st = try todayStage(sc, c.m, c.x, if (i == 0) c.cpre0 else c.last, &ly);
        const h = st[0];
        var deps: [3]A = undefined;
        var n: usize = 0;
        var last = try ly.q_kv.?.apply(sc, h);
        if (ly.idx) |*iw| {
            deps[n] = last;
            n += 1;
            last = try iw.weights_proj.apply(sc, h);
        }
        if (ly.comp) |*cw| {
            deps[n] = last;
            n += 1;
            last = if (ly.ratio > 1) try v41.denseF32x(sc, c.m, &cw.wkv, try asF32(sc, h)) else try cw.wkv.apply(sc, h);
        }
        const dv = mlx.mlx_vector_array_new_data(&deps, n);
        defer _ = mlx.mlx_vector_array_free(dv);
        const iv = mlx.mlx_vector_array_new_data(&[_]A{last}, 1);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_depends(&ov, iv, dv));
        c.last = try output(sc, ov, 0);
        return c.last;
    }
};

/// The bytes one launch reads: q_kv with its scales, the mix weights, weights_proj and the compressor.
fn layerBytes(p: Plan) usize {
    return p.nq * p.dim * 33 / 32 + n_mix * hc * p.dim * 4 + (p.nw + p.nc) * p.dim * 2;
}

test "dsv41 dec in_proj: ubench against its byte floor and the chain it replaces (DSV41_DEC_UBENCH)" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const k = try Kernels.build();
    defer k.deinit();
    const kinds = [_]struct { lo: usize, hi: usize, li: usize, name: []const u8 }{
        .{ .lo = 20, .hi = 25, .li = 21, .name = "plain" },
        .{ .lo = 20, .hi = 25, .li = 24, .name = "index" },
        .{ .lo = 20, .hi = 25, .li = 20, .name = "kv ratio 1" },
        .{ .lo = 2, .hi = 4, .li = 2, .name = "kv ratio 2" },
    };
    var fx: ?*fixture.Fixture = null;
    defer if (fx) |f| f.deinit();
    var fx_lo: usize = 0;
    for (kinds) |kind| {
        if (fx == null or fx_lo != kind.lo) {
            if (fx) |f| f.deinit();
            fx = try fixture.Fixture.init(gpa, .{ .lo = kind.lo, .hi = kind.hi, .ctx = 0 });
            fx_lo = kind.lo;
        }
        const m = fx.?.m;
        const ly = fx.?.layer(kind.li);
        for ([_]usize{ 1, 2, 4, 6 }) |rows| {
            const x = try fx.?.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const cpre = try fx.?.cpre(rows);
            defer _ = mlx.mlx_array_free(cpre);
            var sc = v41.Scope.init(gpa, m.s);
            defer sc.deinit();
            const p = plan(m, ly, x) orelse return error.Declined;
            const base = argsOf(ly, x, cpre, try v41.hcEps(&sc, m));
            const bytes = layerBytes(p);
            const cyc = try Cycle.init(gpa, &sc, ly, bytes, 2_200_000_000);
            defer cyc.deinit(gpa);
            var own_new = v41.Scope.init(gpa, m.s);
            defer own_new.deinit();
            var own_old = v41.Scope.init(gpa, m.s);
            defer own_old.deinit();
            var fresh: Chain = .{ .sc = &own_new, .k = &k, .p = p, .cyc = &cyc, .base = base };
            _ = try decode.chainUs(&fresh, cyc.n);
            const us = try decode.chainUs(&fresh, cyc.n);
            var old: OldChain = .{ .sc = &own_old, .m = m, .ly = ly, .cyc = &cyc, .x = x, .cpre0 = cpre };
            const chain_us = try decode.chainUs(&old, cyc.n);
            var name_buf: [64]u8 = undefined;
            decode.report(try std.fmt.bufPrint(&name_buf, "in_proj ({s}, {d} copies)", .{ kind.name, cyc.n }), rows, us, @as(f64, @floatFromInt(bytes)), chain_us);
        }
    }
}
