//! sdk/blocks/rope/qk_norm_rope.
//! Owns QkAngleSlot, qk_fused_kernel, qk_fused_engaged, qk_fused_override, qk_fused_env, qkNormRopeFusedEnabled, and 18 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const mlxSettledActiveBytes = @import("../../testing/parity.zig").mlxSettledActiveBytes;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const gdnParityRand = @import("../gdn/decode_fused.zig").gdnParityRand;

pub const QkAngleSlot = struct {
    offset: c_int = -1,
    rd: c_int = 0,
    rows: c_int = 0,
    scale: f32 = 1.0,
    arr: mlx.mlx_array = .{ .ctx = null },
};

// ── Fused decode QK-norm+RoPE (mlxfast 9e06de6 class) ──
//
// At decode the chain q/k reshape → per-head RMSNorm → transpose → RoPE
// (→ YaRN mscale on Laguna full layers) is strictly serial — SDPA needs its
// output and nothing overlaps it — and costs 4-6 dispatches per layer for a
// few KB of work. One 32-thread-per-head kernel does all of it. cos/sin are
// never re-derived: a per-token PROBE row (ones|zeros through the stock
// `mlx_fast_rope` at the token's offset) extracts the exact floats the stock
// kernel would use, cached per rope family for the forward (2 probe
// dispatches replace ~160 removed ones). Bit-identity with the composed
// chain is the acceptance bar (`qk norm rope fused` parity tests).
var qk_fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk_fused_engaged: bool = false;
pub var qk_fused_override: ?bool = null;

pub fn qkNormRopeFusedEnabled() bool {
    return qk_fused_override orelse true;
}

// The RMS half mirrors `rms_single_row` at axis 128 / N_READS 4 / one
// simdgroup (lane l owns [4l, 4l+4), f32 squares in index order, `simd_sum`,
// `precise::rsqrt`); the T(x*inv) rounding inside the w-multiply is the value
// the separate kernel would have written and the rotation would have read
// back. The rotation mirrors `rope_single_impl<T, false>`: pair p couples
// elements p and p+RD/2, partner values arrive by simd_shuffle (RD/8 lanes
// apart), f32 arithmetic, one T rounding — then the optional mscale multiply
// reproduces the composed path's separate broadcast multiply (one more T
// rounding, table value 1.0 on the pass-through tail).
pub const QK_NORM_ROPE_SOURCE = kernelSource("kernels/rope/qk_norm_rope.metal");

fn getQkNormRopeKernel() !mlx.mlx_fast_metal_kernel {
    if (qk_fused_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "qw", "kw", "angles", "ms", "eps" };
    const output_names = [_][*:0]const u8{ "oq", "ok" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_qk_norm_rope",
        in_vec,
        out_vec,
        QK_NORM_ROPE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qk_fused_kernel = kernel;
    return kernel;
}

const QkFusedCfgKey = struct { hq: c_int, hk: c_int, rd: c_int, scaled: bool, dtype: mlx.mlx_dtype };
/// Two configs: NoPE (rd 0) and rotary layers alternate within one forward (Flint, Kolibri-1).
var qk_fused_cfgs: [2]struct { cfg: ?mlx.mlx_fast_metal_kernel_config = null, key: QkFusedCfgKey = std.mem.zeroes(QkFusedCfgKey) } = .{ .{}, .{} };
var qk_fused_next: usize = 0;

/// True when the fused QK-norm+RoPE kernel is on — exported so an out-of-file
/// caller (tts.zig's talker) can skip building the angles row when it is not.
pub fn qkNormRopeFusedWanted() bool {
    return qkNormRopeFusedEnabled();
}

/// One fused dispatch: per-head RMSNorm + RoPE (+ optional mscale) for q AND
/// k, emitting the transposed [1,H,1,128] layout attention consumes (at L==1
/// that is the flat buffer relabeled, so the transposes vanish too).
/// `angles` is the [RD] f32 cos|sin row from `ropeAngleRow`. Null → caller
/// keeps the composed chain. Shared with tts.zig's talker decode (hd 128).
pub fn fusedQkNormRope(
    s: mlx.mlx_stream,
    q_flat: mlx.mlx_array, // [1, 1, HQ*128]
    k_flat: mlx.mlx_array, // [1, 1, HK*128]
    q_norm_w: mlx.mlx_array,
    k_norm_w: mlx.mlx_array,
    angles: mlx.mlx_array,
    ms: ?mlx.mlx_array, // bf16 [128] mscale table, null = unscaled
    eps_arr: mlx.mlx_array, // 0-dim f32 SCALAR (binds `constant float&`; a [1] array binds a pointer and the kernel fails to compile)
    hq: c_int,
    hk: c_int,
    rd: c_int,
) !?[2]mlx.mlx_array {
    if (!qkNormRopeFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(q_flat);
    if (dt != .bfloat16 or mlx.mlx_array_dtype(k_flat) != .bfloat16) return null;
    // rd=32 is the qwen3.5/3.6 partial-rotary geometry (0.25 x 128); the
    // shuffle mapping (lane ^ (half_rd/4)) is exact for all three widths.
    // rd=0 is a NoPE layer: every element takes the pass-through branch, `angles` is never read.
    if (rd != 0 and rd != 32 and rd != 64 and rd != 128) return null;
    if (q_norm_w.ctx == null or k_norm_w.ctx == null) return null;

    const key = QkFusedCfgKey{ .hq = hq, .hk = hk, .rd = rd, .scaled = ms != null, .dtype = dt };
    const slot = for (&qk_fused_cfgs) |*c| {
        if (c.cfg != null and std.meta.eql(c.key, key)) break c;
    } else blk: {
        const c = &qk_fused_cfgs[qk_fused_next];
        qk_fused_next ^= 1;
        if (c.cfg) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
        c.cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const oq_shape = [_]c_int{ 1, hq, 1, 128 };
        const ok_shape = [_]c_int{ 1, hk, 1, 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oq_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ok_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk) * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SCALED", if (ms != null) 1 else 0));
        c.* = .{ .cfg = config, .key = key };
        break :blk c;
    };

    const kernel = try getQkNormRopeKernel();
    // The unscaled variant never reads `ms` (SCALED == 0 guards the index),
    // but the input slot must be bound — the norm weight stands in.
    const inputs_arr = [_]mlx.mlx_array{ q_flat, k_flat, q_norm_w, k_norm_w, angles, ms orelse q_norm_w, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, slot.cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var oq = mlx.mlx_array_new();
    var ok = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&oq, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ok, outputs_vec, 1));
    if (!qk_fused_engaged) {
        qk_fused_engaged = true;
        log.info("[attn] fused QK-norm+RoPE engaged: hq={d} hk={d} rd={d} scaled={}\n", .{ hq, hk, rd, ms != null });
    }
    return .{ oq, ok };
}

/// Extract the exact cos|sin row the stock RoPE kernel computes at `offset`:
/// a [1,1,1,rd] f32 probe with ones in the first half rotates to
/// [cos(theta_p) | sin(theta_p)] — the same floats, not a re-derivation (the
/// angle math inside `mlx_fast_rope` is float regardless of the tensor
/// dtype, so the f32 probe reads it out exactly).
pub fn ropeAngleRow(
    s: mlx.mlx_stream,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    allocator: std.mem.Allocator,
) !mlx.mlx_array {
    return ropeAngleRows(s, rd, base, freqs, offset, 1, allocator, 1.0, 1.0);
}

/// Multi-position variant: [rows * rd] f32, row p = the cos|sin row at
/// position `offset + p` — the probe is a [1,1,rows,rd] stack of ones|zeros
/// rows, so `mlx_fast_rope` stamps each row at its own position exactly as it
/// would the real tensor.
///
/// `mscale` multiplies the rows (YaRN's attention factor; 1.0 leaves the probe
/// exactly as the stock kernel computed it). Both halves of a rotary pair tile
/// the same frequency, so scaling the rows scales the rotation of the pair —
/// the same product the reference gets from `cos*af, sin*af`.
pub fn ropeAngleRows(
    s: mlx.mlx_stream,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    rows: c_int,
    allocator: std.mem.Allocator,
    mscale: f32,
    scale: f32,
) !mlx.mlx_array {
    const n: usize = @intCast(rd);
    const nrows: usize = @intCast(rows);
    const buf = try allocator.alloc(f32, n * nrows);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = if ((i % n) < n / 2) 1.0 else 0.0;
    const shape = [_]c_int{ 1, 1, rows, rd };
    const probe = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(probe);
    var rotated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rotated);
    try mlx.check(mlx.mlx_fast_rope(&rotated, probe, rd, false, base, scale, offset, freqs, s));
    if (mscale == 1.0) {
        var flat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{rows * rd};
        try mlx.check(mlx.mlx_reshape(&flat, rotated, &flat_shape, 1, s));
        return flat;
    }
    // `defer`, not `errdefer`: the reshape below is a view, so the handle is ours to drop on the
    // success path too (an errdefer leaked one array per decode tick to the MTLSharedEvent cap).
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    const ms = mlx.mlx_array_new_float(mscale);
    defer _ = mlx.mlx_array_free(ms);
    try mlx.check(mlx.mlx_multiply(&scaled, rotated, ms, s));
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{rows * rd};
    try mlx.check(mlx.mlx_reshape(&flat, scaled, &flat_shape, 1, s));
    return flat;
}

// ── Fused QK-norm+RoPE, hd-256 (mlxfast-challenge qwen35_attention_qk_rms_rope port) ──
//
// The hd-128 kernel above never fires on qwen3.5/3.6 full-attention layers
// (every checkpoint in the family is hd 256), so their decode AND verify
// widths paid the composed per-head RMSNorm → transpose → partial-RoPE chain.
// This is the 256-wide sibling: 64 threads per (head, position) row, RMS via
// the same simdgroup-partials tree `rms_single_row` uses, normalized row
// staged bf16 in threadgroup memory (one rounding — the value the separate
// rms kernel would have written), then the RD-dim rotation reads the staged
// row against a PER-POSITION angle table (`ropeAngleRows` probe — the stock
// kernel's own floats, never re-derived; the challenge computes cos/sin
// in-kernel, which trusts JIT transcendentals our house rule forbids).
// Output is head-major [B,H,S,D], so the transposes vanish. Covers S 1..32
// (verify widths + one-chunk tiny-prompt prefills; challenge raised the same
// gate 16 -> 32 in 033f622).
// Bit-identity with the composed chain is the acceptance bar.
pub const QK_NORM_ROPE_256_SOURCE = kernelSource("kernels/rope/qk_norm_rope_256.metal");

var qk256_fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk256_fused_engaged: bool = false;

fn getQkNormRope256Kernel() !mlx.mlx_fast_metal_kernel {
    if (qk256_fused_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "qw", "kw", "vw", "angles", "eps" };
    const output_names = [_][*:0]const u8{ "oq", "ok", "ov" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_qk_norm_rope_256",
        in_vec,
        out_vec,
        QK_NORM_ROPE_256_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qk256_fused_kernel = kernel;
    return kernel;
}

const Qk256CfgKey = struct { hq: c_int, hk: c_int, hv: c_int, seq: c_int, rd: c_int, normed: bool, dtype: mlx.mlx_dtype };
var qk256_fused_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qk256_fused_cfg_key: Qk256CfgKey = std.mem.zeroes(Qk256CfgKey);

/// One fused dispatch for hd-256 heads at S in 1..32: per-head RMSNorm +
/// partial RoPE for q AND k (and, when `v_in`/`v_norm_w` are bound, the
/// norm-only v of Gemma), emitting the transposed [1,H,S,256] layout. Null
/// q/k norm weights (both, and no v) mean rotation only (Spark).
/// `angles` is the [S*RD] f32 table from `ropeAngleRows` (row p = position
/// offset+p). Null → caller keeps the composed chain. The third result is
/// null-ctx without v.
pub fn fusedQkNormRope256(
    s: mlx.mlx_stream,
    q_in: mlx.mlx_array, // [1, S, HQ, 256] (view ok — ensure_row_contiguous materializes)
    k_in: mlx.mlx_array, // [1, S, HK, 256]
    v_in: mlx.mlx_array, // [1, S, HV, 256] or null-ctx
    q_norm_w: mlx.mlx_array,
    k_norm_w: mlx.mlx_array,
    v_norm_w: mlx.mlx_array,
    angles: mlx.mlx_array,
    eps_arr: mlx.mlx_array, // 0-dim f32 scalar
    hq: c_int,
    hk: c_int,
    seq: c_int,
    rd: c_int,
) !?[3]mlx.mlx_array {
    if (!qkNormRopeFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    if (seq < 1 or seq > 32) return null;
    const dt = mlx.mlx_array_dtype(q_in);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k_in) != dt) return null;
    const normed = q_norm_w.ctx != null;
    if (normed != (k_norm_w.ctx != null)) return null;
    if (normed and (mlx.mlx_array_dtype(q_norm_w) != dt or mlx.mlx_array_dtype(k_norm_w) != dt)) return null;
    // The shuffle-free staged rotation is exact for any even rd <= 256 whose
    // pair partner stays inside the row (qwen3.5's 64, Gemma's 256).
    if (rd != 32 and rd != 64 and rd != 128 and rd != 256) return null;
    const with_v = v_in.ctx != null;
    if (with_v and !normed) return null;
    var hv: c_int = 0;
    if (with_v) {
        if (v_norm_w.ctx == null or mlx.mlx_array_dtype(v_in) != dt or mlx.mlx_array_dtype(v_norm_w) != dt) return null;
        const vsh = mlx.getShape(v_in);
        if (vsh.len != 4 or vsh[0] != 1 or vsh[1] != seq or vsh[3] != 256) return null;
        hv = vsh[2];
    }

    const key = Qk256CfgKey{ .hq = hq, .hk = hk, .hv = hv, .seq = seq, .rd = rd, .normed = normed, .dtype = dt };
    if (qk256_fused_cfg == null or !std.meta.eql(qk256_fused_cfg_key, key)) {
        if (qk256_fused_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const oq_shape = [_]c_int{ 1, hq, seq, 256 };
        const ok_shape = [_]c_int{ 1, hk, seq, 256 };
        const ov_shape = [_]c_int{ 1, @max(hv, 1), if (with_v) seq else 1, 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oq_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ok_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ov_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk + hv) * seq * 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HK", hk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HV", hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "S", seq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "NORM", normed));
        qk256_fused_cfg = config;
        qk256_fused_cfg_key = key;
    }

    const kernel = try getQkNormRope256Kernel();
    // Unbound stages (no v, no norm) never read their slots; bind k / q so
    // every input is an array.
    const inputs_arr = [_]mlx.mlx_array{ q_in, k_in, if (with_v) v_in else k_in, if (normed) q_norm_w else q_in, if (normed) k_norm_w else k_in, if (with_v) v_norm_w else k_in, angles, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qk256_fused_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
    var outs: [3]mlx.mlx_array = undefined;
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, outputs_vec, i));
    }
    if (!with_v) {
        _ = mlx.mlx_array_free(outs[2]);
        outs[2] = .{ .ctx = null };
    }
    if (!qk256_fused_engaged) {
        qk256_fused_engaged = true;
        log.info("[attn] fused QK-norm+RoPE (hd-256) engaged: hq={d} hk={d} hv={d} S={d} rd={d}\n", .{ hq, hk, hv, seq, rd });
    }
    return outs;
}

// ── tests ──

/// Composed reference for the fused QK-norm+RoPE tests: the EXACT shipped
/// chain (reshape → fast_rms_norm → transpose → fast_rope → optional bf16
/// mscale multiply), returning [1, H, 1, 128].
fn qkComposedRef(
    s: mlx.mlx_stream,
    flat: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    ms: ?mlx.mlx_array,
    h: c_int,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    eps: f32,
) !mlx.mlx_array {
    const shape4 = [_]c_int{ 1, 1, h, 128 };
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, flat, &shape4, 4, s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, r, norm_w, eps, s));
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, normed, &perm, 4, s));
    var roped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&roped, t, rd, false, base, 1.0, offset, freqs, s));
    if (ms) |m| {
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, roped, m, s));
        _ = mlx.mlx_array_free(roped);
        return scaled;
    }
    return roped;
}

test "qk norm rope fused: bit-identical at the qwen geometry (rd=32, strided post-split q)" {
    const s = mlx.gpuStream();
    qk_fused_override = true;
    defer qk_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x40FD);
    const rnd = prng.random();

    const hq: c_int = 8;
    const hk: c_int = 2;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const base = mlx.mlx_optional_float{ .value = 5000000.0, .has_value = true };
    const no_freqs = mlx.mlx_array{ .ctx = null };

    for ([_]c_int{ 0, 4093 }) |offset| {
        // q arrives as the attn_output_gate SPLIT view: [1,1,hq,256] with the
        // first 128 of each 256 being the query half — a strided view, which
        // ensure_row_contiguous must materialize without changing values.
        const qg_shape = [_]c_int{ 1, 1, hq, 256 };
        const q_gate = try attn256RandBf16(rnd, &qg_shape, s);
        defer _ = mlx.mlx_array_free(q_gate);
        var split_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(split_vec);
        try mlx.check(mlx.mlx_split(&split_vec, q_gate, 2, -1, s));
        var q_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_view);
        try mlx.check(mlx.mlx_vector_array_get(&q_view, split_vec, 0)); // [1,1,hq,128] strided

        const k_shape = [_]c_int{ 1, 1, hk * 128 };
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const w_shape = [_]c_int{128};
        const qw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(qw);
        const kw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(kw);

        const angles = try ropeAngleRow(s, 32, base, no_freqs, offset, std.testing.allocator);
        defer _ = mlx.mlx_array_free(angles);
        const pair = (try fusedQkNormRope(s, q_view, k, qw, kw, angles, null, eps_arr, hq, hk, 32)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);

        // Composed reference over the SAME strided view (the shipped chain
        // norms the view directly).
        var qn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qn);
        try mlx.check(mlx.mlx_fast_rms_norm(&qn, q_view, qw, eps, s));
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var qt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, qn, &perm, 4, s));
        var rq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rq);
        try mlx.check(mlx.mlx_fast_rope(&rq, qt, 32, false, base, 1.0, offset, no_freqs, s));

        const rk = try qkComposedRef(s, k, kw, null, hk, 32, base, no_freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rk);

        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], rq, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[1], rk, s));
    }
}

test "qk norm rope fused hd-256: bit-identical at qwen 24q/4kv rd=64, S 1..16 incl. strided q, bf16 and f16" {
    const s = mlx.gpuStream();
    qk_fused_override = true;
    defer qk_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x256A);
    const rnd = prng.random();

    const hq: c_int = 24;
    const hk: c_int = 4;
    const rd: c_int = 64;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const base = mlx.mlx_optional_float{ .value = 10000000.0, .has_value = true };
    const no_freqs = mlx.mlx_array{ .ctx = null };
    const perm = [_]c_int{ 0, 2, 1, 3 };

    for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        for ([_]c_int{ 1, 3, 9, 16, 32 }) |seq| {
            for ([_]c_int{ 0, 8191 }) |offset| {
                // q arrives as the attn_output_gate SPLIT view: [1,S,hq,512] with
                // the first 256 of each 512 being the query half.
                const qg_shape = [_]c_int{ 1, seq, hq, 512 };
                const q_gate = try gdnParityRand(rnd, &qg_shape, 1.0, dt, s);
                defer _ = mlx.mlx_array_free(q_gate);
                var split_vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(split_vec);
                try mlx.check(mlx.mlx_split(&split_vec, q_gate, 2, -1, s));
                var q_view = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(q_view);
                try mlx.check(mlx.mlx_vector_array_get(&q_view, split_vec, 0)); // [1,S,hq,256] strided

                const k_shape = [_]c_int{ 1, seq, hk, 256 };
                const k = try gdnParityRand(rnd, &k_shape, 1.0, dt, s);
                defer _ = mlx.mlx_array_free(k);
                const w_shape = [_]c_int{256};
                const qw = try gdnParityRand(rnd, &w_shape, 1.0, dt, s);
                defer _ = mlx.mlx_array_free(qw);
                const kw = try gdnParityRand(rnd, &w_shape, 1.0, dt, s);
                defer _ = mlx.mlx_array_free(kw);

                const angles = try ropeAngleRows(s, rd, base, no_freqs, offset, seq, std.testing.allocator, 1.0, 1.0);
                defer _ = mlx.mlx_array_free(angles);
                const none_v = mlx.mlx_array{ .ctx = null };
                const pair = (try fusedQkNormRope256(s, q_view, k, none_v, qw, kw, none_v, angles, eps_arr, hq, hk, seq, rd)) orelse return error.FusedDeclined;
                defer _ = mlx.mlx_array_free(pair[0]);
                defer _ = mlx.mlx_array_free(pair[1]);

                // Composed reference over the SAME inputs (norm → transpose → rope).
                inline for (.{ .{ q_view, qw, pair[0] }, .{ k, kw, pair[1] } }) |case| {
                    var n = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(n);
                    try mlx.check(mlx.mlx_fast_rms_norm(&n, case[0], case[1], eps, s));
                    var t = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(t);
                    try mlx.check(mlx.mlx_transpose_axes(&t, n, &perm, 4, s));
                    var r = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(r);
                    try mlx.check(mlx.mlx_fast_rope(&r, t, rd, false, base, 1.0, offset, no_freqs, s));
                    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(case[2], r, s));
                }
            }
        }
    }
}

test "qk norm rope fused hd-256 with v rows: bit-identical at gemma 8q/2kv, rd 256 and 64, scaled rope" {
    const s = mlx.gpuStream();
    qk_fused_override = true;
    defer qk_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x6E4B);
    const rnd = prng.random();
    const hq: c_int = 8;
    const hk: c_int = 2;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const no_freqs = mlx.mlx_array{ .ctx = null };
    const perm = [_]c_int{ 0, 2, 1, 3 };
    for ([_]c_int{ 256, 64 }) |rd| {
        for ([_]f32{ 1.0, 0.125 }) |scale| {
            const base = mlx.mlx_optional_float{ .value = if (rd == 256) 10000.0 else 1000000.0, .has_value = true };
            for ([_]c_int{ 0, 4097 }) |offset| {
                const q_shape = [_]c_int{ 1, 1, hq, 256 };
                const q = try gdnParityRand(rnd, &q_shape, 1.0, .bfloat16, s);
                defer _ = mlx.mlx_array_free(q);
                const kv_shape = [_]c_int{ 1, 1, hk, 256 };
                const k = try gdnParityRand(rnd, &kv_shape, 1.0, .bfloat16, s);
                defer _ = mlx.mlx_array_free(k);
                const v = try gdnParityRand(rnd, &kv_shape, 1.0, .bfloat16, s);
                defer _ = mlx.mlx_array_free(v);
                const w_shape = [_]c_int{256};
                const qw = try gdnParityRand(rnd, &w_shape, 1.0, .bfloat16, s);
                defer _ = mlx.mlx_array_free(qw);
                const kw = try gdnParityRand(rnd, &w_shape, 1.0, .bfloat16, s);
                defer _ = mlx.mlx_array_free(kw);
                const one = bf16Scalar(1.0, s);
                defer _ = mlx.mlx_array_free(one);
                var vw = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(vw);
                try mlx.check(mlx.mlx_full(&vw, &w_shape, 1, one, .bfloat16, s));

                const angles = try ropeAngleRows(s, rd, base, no_freqs, offset, 1, std.testing.allocator, 1.0, scale);
                defer _ = mlx.mlx_array_free(angles);
                const got = (try fusedQkNormRope256(s, q, k, v, qw, kw, vw, angles, eps_arr, hq, hk, 1, rd)) orelse return error.FusedDeclined;
                defer for (got) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                inline for (.{ .{ q, qw, got[0], true }, .{ k, kw, got[1], true }, .{ v, vw, got[2], false } }) |case| {
                    var n = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(n);
                    try mlx.check(mlx.mlx_fast_rms_norm(&n, case[0], case[1], eps, s));
                    var t = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(t);
                    try mlx.check(mlx.mlx_transpose_axes(&t, n, &perm, 4, s));
                    var r = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(r);
                    if (case[3]) {
                        try mlx.check(mlx.mlx_fast_rope(&r, t, rd, false, base, scale, offset, no_freqs, s));
                    } else {
                        _ = mlx.mlx_array_set(&r, t);
                    }
                    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(case[2], r, s));
                }
            }
        }
    }
}

test "qk norm rope fused: bit-identical to the composed chain (sliding, rd=128)" {
    const s = mlx.gpuStream();
    qk_fused_override = true;
    defer qk_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x40FE);
    const rnd = prng.random();

    const hq: c_int = 8;
    const hk: c_int = 2;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const base = mlx.mlx_optional_float{ .value = 10000.0, .has_value = true };
    const no_freqs = mlx.mlx_array{ .ctx = null };

    for ([_]c_int{ 0, 977, 65533 }) |offset| {
        const q_shape = [_]c_int{ 1, 1, hq * 128 };
        const k_shape = [_]c_int{ 1, 1, hk * 128 };
        const w_shape = [_]c_int{128};
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const qw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(qw);
        const kw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(kw);

        const angles = try ropeAngleRow(s, 128, base, no_freqs, offset, std.testing.allocator);
        defer _ = mlx.mlx_array_free(angles);
        const pair = (try fusedQkNormRope(s, q, k, qw, kw, angles, null, eps_arr, hq, hk, 128)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);

        const rq = try qkComposedRef(s, q, qw, null, hq, 128, base, no_freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rq);
        const rk = try qkComposedRef(s, k, kw, null, hk, 128, base, no_freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rk);

        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], rq, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[1], rk, s));
    }
}

test "qk norm rope fused: rd=0 is the per-head norm alone, bit-identical to rms_norm + transpose (Flint's NoPE layers, 32q/4kv)" {
    const s = mlx.gpuStream();
    qk_fused_override = true;
    defer qk_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x0F17);
    const rnd = prng.random();
    const hq: c_int = 32;
    const hk: c_int = 4;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, hq * 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, hk * 128 }, s);
    defer _ = mlx.mlx_array_free(k);
    const qw = try attn256RandBf16(rnd, &[_]c_int{128}, s);
    defer _ = mlx.mlx_array_free(qw);
    const kw = try attn256RandBf16(rnd, &[_]c_int{128}, s);
    defer _ = mlx.mlx_array_free(kw);
    // No rotation reads the angles row; the norm weight binds in its place.
    const pair = (try fusedQkNormRope(s, q, k, qw, kw, qw, null, eps_arr, hq, hk, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(pair[0]);
    defer _ = mlx.mlx_array_free(pair[1]);
    for ([_]struct { mlx.mlx_array, mlx.mlx_array, c_int, mlx.mlx_array }{ .{ q, qw, hq, pair[0] }, .{ k, kw, hk, pair[1] } }) |c| {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, c[0], &[_]c_int{ 1, 1, c[2], 128 }, 4, s));
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, r, c[1], eps, s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, normed, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(c[3], t, s));
    }
}

test "qk norm rope fused: bit-identical incl. YaRN freqs + mscale (rd=64)" {
    const s = mlx.gpuStream();
    qk_fused_override = true;
    defer qk_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x40FF);
    const rnd = prng.random();

    const hq: c_int = 48;
    const hk: c_int = 8;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);

    // YaRN-shaped freqs vector [rd/2] f32 and a Laguna-shaped mscale table:
    // the YaRN factor on the rotated 64, exactly 1.0 on the pass-through tail.
    var freq_buf: [32]f32 = undefined;
    for (&freq_buf, 0..) |*f, i| f.* = std.math.pow(f32, 500000.0, @as(f32, @floatFromInt(i)) / 32.0);
    const f_shape = [_]c_int{32};
    const freqs = mlx.mlx_array_new_data(&freq_buf, &f_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);
    const no_base = mlx.mlx_optional_float{ .value = 0, .has_value = false };

    var ms_buf: [128]f32 = undefined;
    for (&ms_buf, 0..) |*m, i| m.* = if (i < 64) 1.3465735912322998 else 1.0;
    const ms_shape = [_]c_int{128};
    const ms_f32 = mlx.mlx_array_new_data(&ms_buf, &ms_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(ms_f32);
    var ms_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ms_bf);
    try mlx.check(mlx.mlx_astype(&ms_bf, ms_f32, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(ms_bf));

    for ([_]c_int{ 0, 12345 }) |offset| {
        const q_shape = [_]c_int{ 1, 1, hq * 128 };
        const k_shape = [_]c_int{ 1, 1, hk * 128 };
        const w_shape = [_]c_int{128};
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const qw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(qw);
        const kw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(kw);

        const angles = try ropeAngleRow(s, 64, no_base, freqs, offset, std.testing.allocator);
        defer _ = mlx.mlx_array_free(angles);
        const pair = (try fusedQkNormRope(s, q, k, qw, kw, angles, ms_bf, eps_arr, hq, hk, 64)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);

        const rq = try qkComposedRef(s, q, qw, ms_bf, hq, 64, no_base, freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rq);
        const rk = try qkComposedRef(s, k, kw, ms_bf, hk, 64, no_base, freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rk);

        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], rq, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[1], rk, s));
    }
}

// The YaRN `mscale != 1.0` arm registered only an `errdefer` for its `scaled` intermediate, so
// every successful call leaked one array per decode tick, each pinning its `MTLSharedEvent`,
// until the process wedged at the 65,535 cap. Settled residency is the observable.
test "ropeAngleRows hands back every array it builds, success path included" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var freq_buf: [32]f32 = undefined;
    for (&freq_buf, 0..) |*f, i| f.* = std.math.pow(f32, 10000000.0, @as(f32, @floatFromInt(i)) / 32.0);
    const f_shape = [_]c_int{32};
    const freqs = mlx.mlx_array_new_data(&freq_buf, &f_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);
    try mlx.check(mlx.mlx_array_eval(freqs));
    const no_base = mlx.mlx_optional_float{ .value = 0, .has_value = false };
    // qwen4_exp geometry: head_dim 256 x partial_rotary_factor 0.25.
    const rd: c_int = 64;

    // 1.0 is the unscaled arm; the second value is the YaRN attention factor at `factor: 4.0`.
    for ([_]f32{ 1.0, 1.1386294364929199 }) |mscale| {
        const c0 = mlx.op_count.load(.monotonic);
        const warm = try ropeAngleRows(s, rd, no_base, freqs, 0, 1, allocator, mscale, 1.0);
        const n_ops = mlx.op_count.load(.monotonic) - c0;
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        try testing.expect(n_ops >= 2);

        var k: u64 = 0;
        while (k <= n_ops) : (k += 1) {
            const base = mlxSettledActiveBytes(s);
            mlx.fault.arm(k); // arm(0) is a no-op arm: the SUCCESS path
            const r = ropeAngleRows(s, rd, no_base, freqs, @intCast(k + 1), 1, allocator, mscale, 1.0);
            mlx.fault.disarm();
            if (r) |a| {
                try mlx.check(mlx.mlx_array_eval(a));
                _ = mlx.mlx_array_free(a);
            } else |err| try testing.expectEqual(error.MlxError, err);
            const after = mlxSettledActiveBytes(s);
            if (after != base) std.debug.print(
                "[angle-rows] mscale={d} k={d}: base {d} after {d}\n",
                .{ mscale, k, base, after },
            );
            try testing.expectEqual(base, after);
        }
    }
}
