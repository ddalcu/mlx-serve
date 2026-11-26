//! `MLX_SERVE_DSV41_PF_HC_V2` (on by default; =0 the five passes): a resident wide prompt layer's hyper-connections in three passes over the
//! stream instead of five, each a release-shape kernel (`kernels/hc_v2_mix.metal`, `kernels/hc_v2_fin.metal`).
//!
//! The control reads and writes the `[L, 4, 5120]` bf16 stream in five launches a layer (the attention side's mix and
//! norm, the FFN side's expand and norm, the FFN side's mixes over x1, the tail's expand, each re-read by 4-5 mix
//! threadgroups), and its mix threadgroups re-read `fn` (24 x 20480 f32) every 4 rows. Here:
//! - the FFN side (`ffn`): x1 = post y + comb^T x, its collapse and its 24 mixes from registers; x1 is never stored;
//! - the tail (`tail`): x1 again from the same words, x2 = post2 y2 + comb2^T x1 stored as the layer's stream, and the
//!   NEXT layer's attention-side collapse and mixes of x2 from registers, handed over in `slot` (`take`); a layer
//!   whose stream another stage changes first (an Engram layer) runs `attn` over its own stream instead.
//! Each mix pass writes per-slice partials; `fin` sums them, makes the split and rms-norms the collapse. 64 rows a
//! threadgroup share each 12 KB step of `fn`, gathered from its stored layout into threadgroup memory in the fragments'
//! lane order (`packAt`); no pass converts or copies a weight. The words are the stock kernels' fmas and roundings; only
//! the sums' order differs.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const gather_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const mix_source = kernelSource("plugins/deepseek_v41/kernels/hc_v2_mix.metal");
pub const fin_source = kernelSource("plugins/deepseek_v41/kernels/hc_v2_fin.metal");

/// The release's shapes, the kernels' compile-time constants.
pub const D: usize = 5120;
pub const HC: usize = 4;
pub const MIX: usize = (2 + HC) * HC;
pub const N: usize = HC * D;
/// Columns a step, rows a threadgroup, the partials' width a row, the fin kernel's threads.
pub const DC: usize = 32;
pub const ROWS: usize = 64;
pub const PW: usize = MIX + 2;
pub const FIN_TG: usize = 128;
/// Floats of `fn` staged a step: 16 k-tiles x 3 mix tiles x 64.
pub const FSTEP: usize = HC * DC / 8 * 3 * 64;

/// The mix kernel's variants: the stream as it is, the FFN side's x1, the tail's x2 (with the next layer's mixes, or
/// alone), its MoE rows given (`y2`) or combined in the kernel from the MoE's parts (`_c`: moe_tail.metal's combine).
pub const Pro = enum { plain, ffn, tail, tail_only, tail_c, tail_only_c };
pub const all = [_]Pro{ .plain, .ffn, .tail, .tail_only, .tail_c, .tail_only_c };

pub fn isTail(p: Pro) bool {
    return p != .plain and p != .ffn;
}
pub fn mixes(p: Pro) bool {
    return p != .tail_only and p != .tail_only_c;
}
pub fn combines(p: Pro) bool {
    return p == .tail_c or p == .tail_only_c;
}
pub fn ins(p: Pro) []const [*:0]const u8 {
    return switch (p) {
        .plain => &.{ "x", "cpre", "fn" },
        .ffn => &.{ "x", "y", "post", "comb", "cpre", "fn" },
        .tail => &.{ "x", "y", "post", "comb", "y2", "post2", "comb2", "cpre", "fn" },
        .tail_only => &.{ "x", "y", "post", "comb", "y2", "post2", "comb2" },
        .tail_c => &.{ "x", "y", "post", "comb", "dn", "inv", "sh", "post2", "comb2", "cpre", "fn" },
        .tail_only_c => &.{ "x", "y", "post", "comb", "dn", "inv", "sh", "post2", "comb2" },
    };
}
pub fn outs(p: Pro) []const [*:0]const u8 {
    return switch (p) {
        .plain, .ffn => &.{ "hraw", "part" },
        .tail, .tail_c => &.{ "x2", "hraw", "part" },
        .tail_only, .tail_only_c => &.{"x2"},
    };
}
pub fn header(p: Pro) [:0]const u8 {
    return switch (p) {
        .plain => "#define PRO 0\n#define MIXON 1\n#define COMB 0\n",
        .ffn => "#define PRO 1\n#define MIXON 1\n#define COMB 0\n",
        .tail => "#define PRO 2\n#define MIXON 1\n#define COMB 0\n",
        .tail_only => "#define PRO 2\n#define MIXON 0\n#define COMB 0\n",
        .tail_c => "#define PRO 2\n#define MIXON 1\n#define COMB 1\n",
        .tail_only_c => "#define PRO 2\n#define MIXON 0\n#define COMB 1\n",
    };
}
pub fn name(p: Pro) [:0]const u8 {
    return switch (p) {
        .plain => "dsv41_hc_v2_plain",
        .ffn => "dsv41_hc_v2_ffn",
        .tail => "dsv41_hc_v2_tail",
        .tail_only => "dsv41_hc_v2_tail_only",
        .tail_c => "dsv41_hc_v2_tail_c",
        .tail_only_c => "dsv41_hc_v2_tail_only_c",
    };
}
/// The routes a row the combine sums (the release's top 6).
pub const K: usize = 6;
pub const fin_ins = [_][*:0]const u8{ "part", "hraw", "scale", "base", "w", "eps" };
pub const fin_outs = [_][*:0]const u8{ "z", "pre", "post", "comb", "h" };

/// d-slices a row: enough threadgroups to fill the GPU at every prompt width (160 steps of 32 columns a row).
pub fn slicesFor(L: usize) usize {
    if (L >= 2048) return 10;
    if (L >= 512) return 20;
    return 40;
}

/// The kernels and the attention side handed to the next layer; one copy across `-Dunits` units.
const State = struct {
    k: [all.len]?mlx.mlx_fast_metal_kernel = @splat(null),
    /// The combining tails reading their down rows in pair order (`MoeParts.tok`).
    k_tok: [all.len]?mlx.mlx_fast_metal_kernel = @splat(null),
    fin: ?mlx.mlx_fast_metal_kernel = null,
    logged: bool = false,
    slot: Slot = .{},
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_hc_v2", .{});
inline fn ps() *State {
    return shared_state.get();
}

/// A wide layer's attention-side hc, made by the layer before's tail: the layer, the handle of the stream it belongs
/// to (the spans' own, compared, never owned), the mixes, the split and the normed collapse (handles this slot owns).
const Slot = struct { li: usize = 0, x: ?*anyopaque = null, z: A = none, pre: A = none, post: A = none, comb: A = none, h: A = none };

/// Whether the switch serves this model's wide layers: the resident trunk on a NAX GPU, bf16, the release's hc shapes.
/// NAX only: the kernels' M1/M2 thread limits cannot be compiled offline (every requested build differs), so an older
/// GPU keeps the stock hc launches.
pub fn serves(m: *const dsv41.Dsv41Model) bool {
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_PF_HC_V2) or !device.naxAvailable()) return false;
    return m.streamed == null and m.tp == null and m.trace == null and m.act == .bfloat16 and
        m.cfg.hidden_size == D and m.cfg.dsv4_hc_mult == HC and mlx.streamIsGpu(m.s);
}

fn engaged(L: usize) void {
    if (ps().logged) return;
    ps().logged = true;
    log.info("[dsv41] MLX_SERVE_DSV41_PF_HC_V2 engaged (rows {d}, {d} slices)\n", .{ L, slicesFor(L) });
}

fn kernel(p: Pro) !mlx.mlx_fast_metal_kernel {
    const i = @intFromEnum(p);
    if (ps().k[i]) |k| return k;
    const k = try dsv41.buildKernel(name(p), ins(p), outs(p), mix_source, header(p));
    ps().k[i] = k;
    return k;
}

/// `kernel(p)` for a combining tail whose down rows are in pair order (`TOK`, `MoeParts.tok`).
fn kernelTok(p: Pro) !mlx.mlx_fast_metal_kernel {
    const i = @intFromEnum(p);
    if (ps().k_tok[i]) |k| return k;
    const k = switch (p) {
        inline .tail_c, .tail_only_c => |q| try dsv41.buildKernel(comptime (name(q) ++ "_tok"), ins(q), outs(q), mix_source, comptime (header(q) ++ "#define TOK 1\n")),
        else => return error.HcV2TokNotCombining,
    };
    ps().k_tok[i] = k;
    return k;
}

fn finKernel() !mlx.mlx_fast_metal_kernel {
    if (ps().fin) |k| return k;
    const k = try dsv41.buildKernel("dsv41_hc_v2_fin", &fin_ins, &fin_outs, fin_source, "");
    ps().fin = k;
    return k;
}

/// Staged position `pos` of a step's `FSTEP` floats -> its element of `fn` `[MIX, N]` (the mix kernel's gather):
/// B fragment (k-tile kt, mix tile mt) at lane `lane` holds fn[mt * 8 + mc + v] at the column the A fragment's
/// k = mr carries, where k-tile kt = i * 4 + t of a step holds copy i's columns (k / 2) * 8 + 2t + k % 2.
pub fn packAt(step: usize, pos: usize) u32 {
    const v = pos % 2;
    const lane = pos / 2 % 32;
    const mt = pos / 64 % 3;
    const kt = pos / 192;
    const i = kt / 4;
    const t = kt % 4;
    const qid = lane / 4;
    const mr = (qid & 4) + (lane / 2) % 4;
    const mc = (qid & 2) * 2 + (lane % 2) * 2;
    const col = i * D + step * DC + (mr / 2) * 8 + 2 * t + mr % 2;
    return @intCast((mt * 8 + mc + v) * N + col);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}
fn rows(x: A) usize {
    return @intCast(mlx.getShape(x)[0]);
}

/// One mix-kernel launch over `L` rows: its outputs (x2 for the tail, then hraw and part where it mixes).
pub fn mixLaunch(sc: *dsv41.Scope, p: Pro, in: []const A, L: usize, S: usize) ![3]A {
    return mixLaunchTo(sc, p, in, L, S, false);
}

/// `mixLaunch`; `tok`: a combining tail's down rows are in pair order (`kernelTok`).
pub fn mixLaunchTo(sc: *dsv41.Scope, p: Pro, in: []const A, L: usize, S: usize, tok: bool) ![3]A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (isTail(p)) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), HC, D }, 3, .bfloat16));
    if (mixes(p)) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), D }, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(S), ci(L), PW }, 3, .float32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((L + ROWS - 1) / ROWS * 256), ci(S), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", ci(S)));
    if (combines(p)) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", K));
    const iv = mlx.mlx_vector_array_new_data(in.ptr, in.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, if (tok) try kernelTok(p) else try kernel(p), iv, cfg, sc.s));
    var got: [3]A = .{ none, none, none };
    const n = nOuts(p);
    for (got[0..n], 0..) |*o, k| {
        var a = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&a, ov, k), &a);
    }
    return got;
}

pub fn nOuts(p: Pro) usize {
    return (if (isTail(p)) @as(usize, 1) else 0) + (if (mixes(p)) @as(usize, 2) else 0);
}

pub const Res = struct { z: A, n: dsv41.Normed };

/// The fin kernel: the slices' partials `part` and the raw collapse `hraw` into the mixes, the split and the normed
/// collapse.
pub fn finLaunch(sc: *dsv41.Scope, part: A, hraw: A, hc: *const dsv41.Hc, norm_w: A, eps: A, S: usize, iters: u32) !Res {
    const L = rows(hraw);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), MIX }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), HC }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), HC }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), HC, HC }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), D }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, FIN_TG, ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, FIN_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", ci(S)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", FIN_TG));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ part, hraw, hc.scale, hc.base, norm_w, eps }, 6);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try finKernel(), iv, cfg, sc.s));
    var o: [5]A = undefined;
    for (&o, 0..) |*x, k| {
        var a = mlx.mlx_array_new();
        x.* = try sc.res(mlx.mlx_vector_array_get(&a, ov, k), &a);
    }
    return .{ .z = o[0], .n = .{ .sp = .{ .pre = o[1], .post = o[2], .comb = o[3] }, .h = o[4] } };
}

fn bf16(sc: *dsv41.Scope, x: A) !A {
    return sc.astype(x, .bfloat16);
}

/// The attention side of stream `x` `[L, 4, D]`: its mixes under `hc`, split, and its collapse with `cpre` rms-normed
/// by `norm_w` (`attnHcWide`'s words).
pub fn attn(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, x: A, cpre: A, hc: *const dsv41.Hc, norm_w: A) !Res {
    const L = rows(x);
    const S = slicesFor(L);
    engaged(L);
    const eps = try dsv41.hcEps(sc, m);
    const o = try mixLaunch(sc, .plain, &.{ x, cpre, hc.fn_w }, L, S);
    return finLaunch(sc, o[1], o[0], hc, norm_w, eps, S, m.cfg.dsv4_hc_sinkhorn_iters);
}

/// The FFN side: x1 = post y + comb^T x (`sa`), its mixes under `hc`, split, and its collapse with `sa.pre` rms-normed
/// by `norm_w` (`hcExpandPreNorm`'s words; x1 itself is not made).
pub fn ffn(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, x: A, y: A, sa: *const dsv41.Split, hc: *const dsv41.Hc, norm_w: A) !Res {
    const L = rows(x);
    const S = slicesFor(L);
    engaged(L);
    const eps = try dsv41.hcEps(sc, m);
    const o = try mixLaunch(sc, .ffn, &.{ x, try bf16(sc, y), sa.post, sa.comb, sa.pre, hc.fn_w }, L, S);
    return finLaunch(sc, o[1], o[0], hc, norm_w, eps, S, m.cfg.dsv4_hc_sinkhorn_iters);
}

/// The next layer's attention side, made in the tail: its mix weights and norm.
pub const Next = struct { hc: *const dsv41.Hc, norm_w: A };
pub const Tail = struct { x2: A, next: ?Res };
/// The MoE's output for the tail: its rows `[L, D]`, or its down rows, unsort and shared rows (`moePrefillParts`) for
/// the kernel to combine.
pub const Moe = union(enum) { rows: A, parts: dsv41.MoeParts };

/// The tail: x2 = post2 y2 + comb2^T x1 (`sf`) where x1 = post y + comb^T x (`sa`) and y2 the MoE's output, and with
/// `nx` the next layer's attention side of x2 (its collapse with `sf.pre`, the next layer's previous mix).
pub fn tail(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, x: A, y: A, sa: *const dsv41.Split, y2: Moe, sf: *const dsv41.Split, nx: ?Next) !Tail {
    const L = rows(x);
    const S = slicesFor(L);
    engaged(L);
    const yb = try bf16(sc, y);
    var in: [11]A = undefined;
    var n_in: usize = 0;
    for ([_]A{ x, yb, sa.post, sa.comb }) |a| {
        in[n_in] = a;
        n_in += 1;
    }
    const c = y2 == .parts;
    const tok = c and y2.parts.tok;
    switch (y2) {
        .rows => |r| {
            in[n_in] = try bf16(sc, r);
            n_in += 1;
        },
        .parts => |p| for ([_]A{ try bf16(sc, p.dn), try sc.astype(p.inv, .uint32), try bf16(sc, p.sh) }) |a| {
            in[n_in] = a;
            n_in += 1;
        },
    }
    for ([_]A{ sf.post, sf.comb }) |a| {
        in[n_in] = a;
        n_in += 1;
    }
    const n = nx orelse {
        const o = try mixLaunchTo(sc, if (c) .tail_only_c else .tail_only, in[0..n_in], L, S, tok);
        return .{ .x2 = o[0], .next = null };
    };
    in[n_in] = sf.pre;
    in[n_in + 1] = n.hc.fn_w;
    n_in += 2;
    const eps = try dsv41.hcEps(sc, m);
    const o = try mixLaunchTo(sc, if (c) .tail_c else .tail, in[0..n_in], L, S, tok);
    return .{ .x2 = o[0], .next = try finLaunch(sc, o[2], o[1], n.hc, n.norm_w, eps, S, m.cfg.dsv4_hc_sinkhorn_iters) };
}

fn own(dst: *A, x: A) void {
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = none;
    if (x.ctx == null) return;
    var o = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&o, x);
    dst.* = o;
}

/// Hands layer `li`'s attention side of stream `x` to that layer (`take`), dropping any held before.
pub fn put(li: usize, x: A, r: Res) void {
    drop();
    const s = &ps().slot;
    s.li = li;
    s.x = x.ctx;
    own(&s.z, r.z);
    own(&s.pre, r.n.sp.pre);
    own(&s.post, r.n.sp.post);
    own(&s.comb, r.n.sp.comb);
    own(&s.h, r.n.h);
}

/// Layer `li`'s attention side when the slot holds it for stream `x` (the handles move into `sc`); any other slot is
/// dropped.
pub fn take(sc: *dsv41.Scope, li: usize, x: A) !?Res {
    const s = &ps().slot;
    if (s.x == null) return null;
    if (s.li != li or s.x != x.ctx) {
        drop();
        return null;
    }
    const r: Res = .{ .z = try sc.keep(s.z), .n = .{ .sp = .{ .pre = try sc.keep(s.pre), .post = try sc.keep(s.post), .comb = try sc.keep(s.comb) }, .h = try sc.keep(s.h) } };
    s.* = .{};
    return r;
}

pub fn drop() void {
    const s = &ps().slot;
    for ([_]A{ s.z, s.pre, s.post, s.comb, s.h }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    s.* = .{};
}

test "hc_v2 bench: the combining tails at a prompt width over a served routing, the stock builds reading the served unsort against the TOK builds reading the down rows in pair order (DSV41_BENCH_HC = rows, DSV41_BENCH_HC_ROUTE = a DSV41_ROUTE_DUMP file)" {
    const rows_s = std.mem.span(env.get(.DSV41_BENCH_HC) orelse return error.SkipZigTest);
    const path = env.get(.DSV41_BENCH_HC_ROUTE) orelse return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable()) return error.SkipZigTest;
    const L: usize = try std.fmt.parseInt(usize, rows_s, 10);
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    const rnd = struct {
        fn f(s: *dsv41.Scope, k: A, sh: []const c_int, dt: mlx.mlx_dtype, sd: f32) !A {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, dt, 0, sd, k, s.s));
            return s.keep(r);
        }
    }.f;
    const Li = ci(L);
    const x = try rnd(&sc, key, &.{ Li, HC, D }, .bfloat16, 1);
    const y = try rnd(&sc, key, &.{ Li, D }, .bfloat16, 1);
    const dn = try rnd(&sc, key, &.{ ci(L * K), D }, .bfloat16, 1);
    const sh = try rnd(&sc, key, &.{ Li, D }, .bfloat16, 1);
    const post = try rnd(&sc, key, &.{ Li, HC }, .float32, 0.5);
    const comb = try rnd(&sc, key, &.{ Li, HC, HC }, .float32, 0.25);
    const cpre = try rnd(&sc, key, &.{ Li, HC }, .float32, 0.5);
    const fnw = try rnd(&sc, key, &.{ MIX, N }, .float32, 0.01);
    try dsv41.evalAll(&.{ x, y, dn, sh, post, comb, cpre, fnw });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(path), gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    const S = slicesFor(L);
    // One launch's time: a chain of `n` launches, each reading the one before's x2 as its stream, evaluated at once;
    // the chain of 9 less the chain of 1, over 8, best of 4.
    const Chain = struct {
        fn run(p: Pro, in0: []const A, rows_: usize, S_: usize, n: usize, tok: bool) !f64 {
            const g = std.testing.allocator;
            const io2 = std.Io.Threaded.global_single_threaded.io();
            const iou = @import("../../../sdk/util/io_util.zig");
            var best: f64 = std.math.inf(f64);
            for (0..4) |_| {
                var it = dsv41.Scope.init(g, mlx.gpuStream());
                defer it.deinit();
                var in: [11]A = undefined;
                @memcpy(in[0..in0.len], in0);
                var last: A = none;
                const t0 = iou.nowUsMonotonic(io2);
                for (0..n) |_| {
                    const o = try mixLaunchTo(&it, p, in[0..in0.len], rows_, S_, tok);
                    in[0] = o[0];
                    last = o[0];
                }
                try dsv41.evalAll(&.{last});
                best = @min(best, @as(f64, @floatFromInt(iou.nowUsMonotonic(io2) - t0)));
            }
            return best;
        }
        fn per(p: Pro, in0: []const A, rows_: usize, S_: usize, tok: bool) !f64 {
            return (try run(p, in0, rows_, S_, 9, tok) - try run(p, in0, rows_, S_, 1, tok)) / 8;
        }
    };
    const iv = try gpa.alloc(u32, L * K);
    defer gpa.free(iv);
    const fill = try gpa.alloc(u8, L);
    defer gpa.free(fill);
    var tot: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } };
    var calls: usize = 0;
    var it = gather_nax.RouteDump.init(bytes);
    while (try it.next()) |rc| {
        if (rc.k != K) return error.BadRouteDump;
        const rows_ = rc.rows_of;
        if (rc.rows != L) continue;
        // The served unsort: pair (token r, slot j) at its sorted place, slot j in the order the token's rows come.
        @memset(fill, 0);
        for (rows_, 0..) |r, place| {
            if (r >= L or fill[r] >= K) return error.BadRouteDump;
            iv[r * K + fill[r]] = @intCast(place);
            fill[r] += 1;
        }
        var isc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer isc.deinit();
        const inv = try isc.keep(mlx.mlx_array_new_data(iv.ptr, &[_]c_int{ Li, K }, 2, .uint32));
        for ([_]Pro{ .tail_c, .tail_only_c }, 0..) |p, pi| {
            const in: []const A = if (mixes(p)) &.{ x, y, post, comb, dn, inv, sh, post, comb, cpre, fnw } else &.{ x, y, post, comb, dn, inv, sh, post, comb };
            const us = try Chain.per(p, in, L, S, false);
            const ut = try Chain.per(p, in, L, S, true);
            tot[pi][0] += us;
            tot[pi][1] += ut;
            std.debug.print("[hc_v2 tok] call {d} {s} {d} rows ({d} slices): served unsort {d:.0} us, pair order (TOK) {d:.0} us ({d:.0} us)\n", .{ calls, @tagName(p), L, S, us, ut, ut - us });
        }
        calls += 1;
    }
    if (calls == 0) return error.NoCallOfThisWidth;
    for ([_]Pro{ .tail_c, .tail_only_c }, 0..) |p, pi| std.debug.print("[hc_v2 tok] {s} over {d} calls of {d} rows: served unsort {d:.2} ms, pair order (TOK) {d:.2} ms ({d:.2} ms)\n", .{ @tagName(p), calls, L, tot[pi][0] / 1e3, tot[pi][1] / 1e3, (tot[pi][1] - tot[pi][0]) / 1e3 });
}

test "hc_v2 packing: every element of fn lands once, each step's floats in its own 32 columns of every copy" {
    const seen = try std.testing.allocator.alloc(bool, N * MIX);
    defer std.testing.allocator.free(seen);
    @memset(seen, false);
    for (0..D / DC) |st| for (0..FSTEP) |p| {
        const e = packAt(st, p);
        try std.testing.expect(!seen[e]);
        seen[e] = true;
        const col = e % N;
        try std.testing.expect(col % D / DC == st);
    };
    for (seen) |s| try std.testing.expect(s);
}
