//! `MLX_SERVE_DSV41_PF_HC_V2` (default off): a resident wide prompt layer's hyper-connections in three passes over the
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
//! threadgroup share each 12 KB step of `fn` (packed host-side in the fragments' lane order, `packIndex`). The words
//! are the stock kernels' fmas and roundings; only the sums' order differs.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");

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
/// Floats of packed `fn` a step: 16 k-tiles x 3 mix tiles x 64.
pub const FSTEP: usize = HC * DC / 8 * 3 * 64;

/// The mix kernel's variants: the stream as it is, the FFN side's x1, the tail's x2 (with the next layer's mixes, or
/// alone).
pub const Pro = enum { plain, ffn, tail, tail_only };

pub fn ins(p: Pro) []const [*:0]const u8 {
    return switch (p) {
        .plain => &.{ "x", "cpre", "fnp" },
        .ffn => &.{ "x", "y", "post", "comb", "cpre", "fnp" },
        .tail => &.{ "x", "y", "post", "comb", "y2", "post2", "comb2", "cpre", "fnp" },
        .tail_only => &.{ "x", "y", "post", "comb", "y2", "post2", "comb2" },
    };
}
pub fn outs(p: Pro) []const [*:0]const u8 {
    return switch (p) {
        .plain, .ffn => &.{ "hraw", "part" },
        .tail => &.{ "x2", "hraw", "part" },
        .tail_only => &.{"x2"},
    };
}
pub fn header(p: Pro) [:0]const u8 {
    return switch (p) {
        .plain => "#define PRO 0\n#define MIXON 1\n",
        .ffn => "#define PRO 1\n#define MIXON 1\n",
        .tail => "#define PRO 2\n#define MIXON 1\n",
        .tail_only => "#define PRO 2\n#define MIXON 0\n",
    };
}
pub fn name(p: Pro) [:0]const u8 {
    return switch (p) {
        .plain => "dsv41_hc_v2_plain",
        .ffn => "dsv41_hc_v2_ffn",
        .tail => "dsv41_hc_v2_tail",
        .tail_only => "dsv41_hc_v2_tail_only",
    };
}
pub const fin_ins = [_][*:0]const u8{ "part", "hraw", "scale", "base", "w", "eps" };
pub const fin_outs = [_][*:0]const u8{ "z", "pre", "post", "comb", "h" };

/// d-slices a row: enough threadgroups to fill the GPU at every prompt width (160 steps of 32 columns a row).
pub fn slicesFor(L: usize) usize {
    if (L >= 2048) return 10;
    if (L >= 512) return 20;
    return 40;
}

/// The kernels, `fn`'s packing index and the attention side handed to the next layer; one copy across `-Dunits` units.
const State = struct {
    k: [4]?mlx.mlx_fast_metal_kernel = @splat(null),
    fin: ?mlx.mlx_fast_metal_kernel = null,
    pack: A = none,
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

/// Whether the switch serves this model's wide layers: the resident trunk on the GPU, bf16, the release's hc shapes.
pub fn serves(m: *const dsv41.Dsv41Model) bool {
    if (!env.on(.MLX_SERVE_DSV41_PF_HC_V2)) return false;
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

fn finKernel() !mlx.mlx_fast_metal_kernel {
    if (ps().fin) |k| return k;
    const k = try dsv41.buildKernel("dsv41_hc_v2_fin", &fin_ins, &fin_outs, fin_source, "");
    ps().fin = k;
    return k;
}

/// Packed position `pos` of a step's `FSTEP` floats -> its element of `fn` `[MIX, N]`: B fragment (k-tile kt, mix
/// tile mt) at lane `lane` holds fn[mt * 8 + mc + v] at the column the A fragment's k = mr carries, where k-tile
/// kt = i * 4 + t of a step holds copy i's columns (k / 2) * 8 + 2t + k % 2 (the mix kernel's lane layout).
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

/// The packing index `[N * MIX]` u32 (`packAt` over every step), made once.
fn packIndex() !A {
    if (ps().pack.ctx != null) return ps().pack;
    const a = std.heap.page_allocator;
    const idx = try a.alloc(u32, N * MIX);
    defer a.free(idx);
    for (0..D / DC) |st| for (0..FSTEP) |p| {
        idx[st * FSTEP + p] = packAt(st, p);
    };
    ps().pack = mlx.mlx_array_new_data(idx.ptr, &[_]c_int{@intCast(N * MIX)}, 1, .uint32);
    return ps().pack;
}

/// `fn` in the mix kernel's order, a fresh array a call (2 MB gathered).
pub fn packed(sc: *dsv41.Scope, fn_w: A) !A {
    return sc.take(try sc.reshape(fn_w, &.{-1}), try packIndex(), 0);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}
fn rows(x: A) usize {
    return @intCast(mlx.getShape(x)[0]);
}

/// One mix-kernel launch over `L` rows: its outputs (x2 for the tail, then hraw and part where it mixes).
pub fn mixLaunch(sc: *dsv41.Scope, p: Pro, in: []const A, L: usize, S: usize) ![3]A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (p == .tail or p == .tail_only) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), HC, D }, 3, .bfloat16));
    if (p != .tail_only) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), D }, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(S), ci(L), PW }, 3, .float32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((L + ROWS - 1) / ROWS * 256), ci(S), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", ci(S)));
    const iv = mlx.mlx_vector_array_new_data(in.ptr, in.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(p), iv, cfg, sc.s));
    var got: [3]A = .{ none, none, none };
    const n: usize = switch (p) {
        .plain, .ffn => 2,
        .tail => 3,
        .tail_only => 1,
    };
    for (got[0..n], 0..) |*o, k| {
        var a = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&a, ov, k), &a);
    }
    return got;
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
    const o = try mixLaunch(sc, .plain, &.{ x, cpre, try packed(sc, hc.fn_w) }, L, S);
    return finLaunch(sc, o[1], o[0], hc, norm_w, eps, S, m.cfg.dsv4_hc_sinkhorn_iters);
}

/// The FFN side: x1 = post y + comb^T x (`sa`), its mixes under `hc`, split, and its collapse with `sa.pre` rms-normed
/// by `norm_w` (`hcExpandPreNorm`'s words; x1 itself is not made).
pub fn ffn(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, x: A, y: A, sa: *const dsv41.Split, hc: *const dsv41.Hc, norm_w: A) !Res {
    const L = rows(x);
    const S = slicesFor(L);
    engaged(L);
    const eps = try dsv41.hcEps(sc, m);
    const o = try mixLaunch(sc, .ffn, &.{ x, try bf16(sc, y), sa.post, sa.comb, sa.pre, try packed(sc, hc.fn_w) }, L, S);
    return finLaunch(sc, o[1], o[0], hc, norm_w, eps, S, m.cfg.dsv4_hc_sinkhorn_iters);
}

/// The next layer's attention side, made in the tail: its mix weights and norm.
pub const Next = struct { hc: *const dsv41.Hc, norm_w: A };
pub const Tail = struct { x2: A, next: ?Res };

/// The tail: x2 = post2 y2 + comb2^T x1 (`sf`) where x1 = post y + comb^T x (`sa`), and with `nx` the next layer's
/// attention side of x2 (its collapse with `sf.pre`, the next layer's previous mix).
pub fn tail(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, x: A, y: A, sa: *const dsv41.Split, y2: A, sf: *const dsv41.Split, nx: ?Next) !Tail {
    const L = rows(x);
    const S = slicesFor(L);
    engaged(L);
    const yb = try bf16(sc, y);
    const y2b = try bf16(sc, y2);
    const n = nx orelse {
        const o = try mixLaunch(sc, .tail_only, &.{ x, yb, sa.post, sa.comb, y2b, sf.post, sf.comb }, L, S);
        return .{ .x2 = o[0], .next = null };
    };
    const eps = try dsv41.hcEps(sc, m);
    const o = try mixLaunch(sc, .tail, &.{ x, yb, sa.post, sa.comb, y2b, sf.post, sf.comb, sf.pre, try packed(sc, n.hc.fn_w) }, L, S);
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
