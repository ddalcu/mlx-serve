//! A KDA layer's prompt prologue as one launch (MLX_SERVE_KDA_PROMPT_PROLOGUE): the chain's depthwise conv, SiLU, q's
//! and k's rms_norm and scales, and v, each the MLX kernel the chain runs in its operand order and types, the sigmoid
//! with the op's own precise exp; the conv window as conv1d.run keeps it. One simdgroup a (row, section, head), eight a
//! threadgroup. It takes GLM's prompt rows where the per-core route declines.
//! copied_from: turboexl3_glm53f src/kda_prompt.zig (`prologueServes`, `prologueText`, `windowOf`) and
//! src/kernels/kda_prologue.metal, kda_prompt_header.metal (G2's TURBOEXL3_GLM53F_KDA_PROLOGUE).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const log = @import("../../sdk/util/log.zig");
const PREFILL_EVAL_MIN_SEQ = @import("../../sdk/bases/run.zig").PREFILL_EVAL_MIN_SEQ;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const depthwise = @import("../../sdk/blocks/conv/depthwise.zig");

const Array = mlx.mlx_array;

const SOURCE = kernelSource("plugins/glm5_next/kernels/kda_prologue.metal");
const HEADER = kernelSource("plugins/glm5_next/kernels/kda_prologue_header.metal");

/// Simdgroups a threadgroup.
const groups: c_int = 8;
/// The head width the text is written for (rms_single_row's 32 lanes x 4 reads).
const head_dim: c_int = 128;

/// What the chain leaves for the recurrence: q and k normed and scaled and v, each [1, rows, heads, 128] contiguous,
/// and the conv window the layer keeps; each a new array.
pub const Prologue = struct { q: Array, k: Array, v: Array, window: Array };

/// Launches since the load (the tests read that the text ran).
pub var runs: u64 = 0;
var engaged_said = false;
var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernelOf() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "qkv", "prev", "w", "q_scale", "k_scale", "eps" };
    const outs = [_][*:0]const u8{ "q_out", "k_out", "v_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("msv_glm_kda_prologue", iv, ov, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// Whether the text takes one slot's qkv rows of `shape` ([1, rows, 3 x hk x dk]) at `hk` x `dk` key and `hv` x `dv`
/// value heads and `taps` conv taps: heads alike and 128 wide, whole threadgroups of (section, head) simdgroups.
pub fn serves(shape: []const c_int, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) bool {
    if (hk != hv or dk != dv or dk != head_dim or hk < 1 or taps < 2) return false;
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != 3 * hk * dk) return false;
    return @rem(3 * hk, groups) == 0;
}

/// The chain's prologue over `qkv` after `prev` (the layer's conv window; null: a cold layer, zeros as the chain takes
/// them), `conv_w` [cdim, taps, 1], the chain's own 0-dim scales and its rms epsilon. Null off the GPU or on a shape or
/// dtype the text does not take (the caller runs the chain).
pub fn prologue(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) !?Prologue {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(qkv);
    if (!serves(sh, hk, dk, hv, dv, taps)) return null;
    const rows = sh[1];
    const cdim = sh[2];
    const dt = mlx.mlx_array_dtype(qkv);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    // The chain's conv weights, window and scales in qkv's dtype (else MLX promotes and the text is not the chain).
    const wsh = mlx.getShape(conv_w);
    if (mlx.mlx_array_dtype(conv_w) != dt or wsh.len != 3 or wsh[0] != cdim or wsh[1] != taps or wsh[2] != 1) return null;
    for ([_]Array{ q_scale, k_scale }) |x| if (mlx.mlx_array_dtype(x) != dt or mlx.mlx_array_ndim(x) != 0) return null;
    if (prev) |p| {
        const psh = mlx.getShape(p);
        if (psh.len != 3 or psh[0] != 1 or psh[1] != taps - 1 or psh[2] != cdim or mlx.mlx_array_dtype(p) != dt) return null;
    }
    var cold: Array = .{ .ctx = null };
    defer if (cold.ctx != null) {
        _ = mlx.mlx_array_free(cold);
    };
    const before = prev orelse blk: {
        cold = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&cold, &[_]c_int{ 1, taps - 1, cdim }, 3, dt, s));
        break :blk cold;
    };
    const window = try windowOf(s, before, qkv, rows, cdim, taps);
    errdefer _ = mlx.mlx_array_free(window);

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{ 1, rows, hk, dk };
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 3 * hk, rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, groups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "DK", dk }, .{ "H", hk }, .{ "DIM", hk * dk }, .{ "CDIM", cdim }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const ins = [_]Array{ qkv, before, conv_w, q_scale, k_scale, eps_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernelOf(), iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 3) return error.MetalKernelBadOutputCount;
    var outs: [3]Array = @splat(.{ .ctx = null });
    errdefer for (outs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
    runs += 1;
    if (!engaged_said) {
        engaged_said = true;
        log.info("[kda] prompt prologue engaged: S={d} Hk={d} Hv={d} cold={}\n", .{ rows, hk, hv, prev == null });
    }
    return .{ .q = outs[0], .k = outs[1], .v = outs[2], .window = window };
}

/// conv1d.run's window over [before | qkv]: its rows [rows, rows + taps - 1), an owned copy past PREFILL_EVAL_MIN_SEQ
/// rows where the chain makes one (`materializedOwnedCopy`), else a slice.
fn windowOf(s: mlx.mlx_stream, before: Array, qkv: Array, rows: c_int, cdim: c_int, taps: c_int) !Array {
    const keep = taps - 1;
    var tail = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tail);
    if (rows >= keep) {
        try mlx.check(mlx.mlx_slice(&tail, qkv, &[_]c_int{ 0, rows - keep, 0 }, 3, &[_]c_int{ 1, rows, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    } else {
        const parts = [_]Array{ before, qkv };
        const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        try mlx.check(mlx.mlx_slice(&tail, cat, &[_]c_int{ 0, rows, 0 }, 3, &[_]c_int{ 1, rows + keep, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    }
    if (depthwise.compactConvStateEnabled() and rows >= PREFILL_EVAL_MIN_SEQ) {
        const owned = try materializedOwnedCopy(s, tail);
        _ = mlx.mlx_array_free(tail);
        depthwise.compact_conv_state_copy_count += 1;
        return owned;
    }
    return tail;
}

// ── tests ──

const testing = std.testing;
const ops = @import("../../mlx/ops.zig");
const conv1d = @import("../../sdk/bases/blocks/conv1d.zig");

/// GLM-5.3-Flash's served KDA layer: 64 key and value heads of 128 (conv width 24576), 4 taps, the rms epsilon.
const t_heads: c_int = 64;
const t_cdim: c_int = 3 * t_heads * head_dim;
const t_eps: f32 = 1e-6 / 128.0;

fn freeA(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Seeded normal bf16 values at `scale`, evaluated.
fn normal(s: mlx.mlx_stream, seed: u64, shape: []const c_int, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, .bfloat16, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// `v` as a 0-dim bf16 scalar (the chain's own scale type).
fn scalarBf16(s: mlx.mlx_stream, v: f32) !Array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    return out;
}

/// The chain kda.forward runs for prompt rows without a launch: conv1d.run (window, depthwise conv, SiLU), q's and k's
/// rms_norm over ones then the scale, v; and the window it keeps. Each a new array.
fn chain(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, qs: Array, ks: Array) ![4]Array {
    const rows = mlx.getShape(qkv)[1];
    const kd = t_heads * head_dim;
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const conv = try conv1d.run(&g, qkv, prev, conv_w, null, 1, t_cdim, 4, true);
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{head_dim}, 1, .bfloat16, s));
    var out: [4]Array = @splat(.{ .ctx = null });
    errdefer for (out) |a| freeA(a);
    for ([_][2]c_int{ .{ 0, kd }, .{ kd, 2 * kd } }, 0..) |span, i| {
        const part = try g.reshape(try g.slice(conv.y, &.{ 0, 0, span[0] }, &.{ 1, rows, span[1] }), &.{ 1, rows, t_heads, head_dim });
        var nrm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nrm);
        try mlx.check(mlx.mlx_fast_rms_norm(&nrm, part, ones, t_eps, s));
        out[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out[i], nrm, if (i == 0) qs else ks, s));
    }
    out[2] = g.escape(try g.reshape(try g.slice(conv.y, &.{ 0, 0, 2 * kd }, &.{ 1, rows, t_cdim }), &.{ 1, rows, t_heads, head_dim }));
    out[3] = g.escape(conv.window);
    try evalAll(&out);
    return out;
}

/// Elements (2 bytes each) of `want` and `got` whose bits differ, both made contiguous; the shapes must match.
fn bitsApart(s: mlx.mlx_stream, want: Array, got: Array) !usize {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| freeA(a);
    for ([_]Array{ want, got }, &parts) |a, *p| {
        p.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(p, a, false, s));
    }
    try evalAll(&parts);
    const n = mlx.mlx_array_size(parts[0]);
    const wb = mlx.mlx_array_data_bfloat16(parts[0]) orelse return error.TestNoData;
    const gb = mlx.mlx_array_data_bfloat16(parts[1]) orelse return error.TestNoData;
    var apart: usize = 0;
    for (0..n) |i| {
        if (wb[i] != gb[i]) apart += 1;
    }
    return apart;
}

/// The launch against the chain on `qkv` after `prev`: q, k, v and the window, bit for bit. Returns the window.
fn expectChain(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, qs: Array, ks: Array) !Array {
    const want = try chain(s, qkv, prev, conv_w, qs, ks);
    defer for (want[0..3]) |a| freeA(a);
    errdefer freeA(want[3]);
    const before = runs;
    const got = (try prologue(s, qkv, prev, conv_w, qs, ks, t_eps, t_heads, head_dim, t_heads, head_dim, 4)) orelse return error.TestPrologueDeclined;
    defer for ([_]Array{ got.q, got.k, got.v, got.window }) |a| freeA(a);
    try testing.expectEqual(before + 1, runs);
    for ([_]Array{ got.q, got.k, got.v, got.window }, want) |g, w| try testing.expectEqual(@as(usize, 0), try bitsApart(s, w, g));
    return want[3];
}

// Bar: at the served shape the launch gives the chain's q, k, v and window bit for bit: a cold 16,356-row chunk, then
// a warm 31-row chunk after its window (the split prompt forward's two spans).
test "glm5_next kda prologue: 64 heads, a cold 16,356-row chunk then a warm 31-row chunk give the chain's bits" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const conv_w = try normal(s, 20261113, &.{ t_cdim, 4, 1 }, 0.5);
    defer freeA(conv_w);
    const qs = try scalarBf16(s, 1.0 / 128.0);
    defer freeA(qs);
    const ks = try scalarBf16(s, @sqrt(1.0 / 128.0));
    defer freeA(ks);
    const a = try normal(s, 20261114, &.{ 1, 16_356, t_cdim }, 1.0);
    defer freeA(a);
    const window = try expectChain(s, a, null, conv_w, qs, ks);
    defer freeA(window);
    const b = try normal(s, 20261115, &.{ 1, 31, t_cdim }, 1.0);
    defer freeA(b);
    const w2 = try expectChain(s, b, window, conv_w, qs, ks);
    freeA(w2);
}

// Bar: the conv's SiLU at every finite bf16 pattern (the newest tap alone, so each conv output is its input) gives the
// chain's bits; the planted control, the same formula with a bare metal::exp (the fast exp in a runtime-compiled
// kernel), misses mlx_sigmoid somewhere, or the sweep could not see a fast exp.
test "glm5_next kda prologue: the SiLU at every finite bf16 pattern gives the chain's bits; the bare-exp formula misses some" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = testing.allocator;
    const rows: c_int = 31;
    const n: usize = @intCast(rows * t_cdim);
    const vals = try a.alloc(f32, n);
    defer a.free(vals);
    var p: u32 = 0;
    var k: usize = 0;
    var finite: usize = 0;
    while (k < n) : (p = (p + 1) & 0xffff) {
        if ((p >> 7) & 0xff == 0xff) continue;
        vals[k] = @bitCast(p << 16);
        k += 1;
        if (k <= 65_280) finite = k;
    }
    try testing.expectEqual(@as(usize, 65_280), finite);
    const shape = [_]c_int{ 1, rows, t_cdim };
    const v32 = mlx.mlx_array_new_data(vals.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(v32);
    var qkv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qkv);
    try mlx.check(mlx.mlx_astype(&qkv, v32, .bfloat16, s));
    const wv = try a.alloc(f32, @intCast(t_cdim * 4));
    defer a.free(wv);
    for (wv, 0..) |*x, i| x.* = if (i % 4 == 3) 1.0 else 0.0;
    const wshape = [_]c_int{ t_cdim, 4, 1 };
    const w32 = mlx.mlx_array_new_data(wv.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(w32);
    var conv_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_w);
    try mlx.check(mlx.mlx_astype(&conv_w, w32, .bfloat16, s));
    try evalAll(&.{ qkv, conv_w });
    const qs = try scalarBf16(s, 1.0 / 128.0);
    defer freeA(qs);
    const ks = try scalarBf16(s, @sqrt(1.0 / 128.0));
    defer freeA(ks);
    const window = try expectChain(s, qkv, null, conv_w, qs, ks);
    freeA(window);

    // The control: MLX's sigmoid vs the formula with a bare metal::exp over the same patterns.
    const src =
        \\const uint i = thread_position_in_grid.x;
        \\const T x = inp[i];
        \\T sy = T(1) / (T(1) + metal::exp(metal::abs(x)));
        \\out[i] = x < T(0) ? sy : T(1) - sy;
    ;
    const in_names = [_][*:0]const u8{"inp"};
    const out_names = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&in_names, 1);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&out_names, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const kern = mlx.mlx_fast_metal_kernel_new("msv_glm_kda_prologue_test_bare_exp", iv, ov, src, "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, qkv, &[_]c_int{@intCast(n)}, 1, s));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@intCast(n)}, 1, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(n), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]Array{flat};
    const ivec = mlx.mlx_vector_array_new_data(&ins, 1);
    defer _ = mlx.mlx_vector_array_free(ivec);
    var ovec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ovec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ovec, kern, ivec, cfg, s));
    var bare = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bare);
    try mlx.check(mlx.mlx_vector_array_get(&bare, ovec, 0));
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_sigmoid(&want, flat, s));
    const missed = try bitsApart(s, want, bare);
    std.debug.print("kda prologue sweep: the bare-exp formula missed mlx_sigmoid at {d} of {d} values\n", .{ missed, n });
    try testing.expect(missed > 0);
}
