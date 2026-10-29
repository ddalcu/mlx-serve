//! sdk/blocks/gdn/decode_fused.
//! Owns GDN_KERNEL_HEADER, GDN_PREWORK_SOURCE, gdn_prework_kernel, gdn_prework_engaged, gdn_prework_override, gdn_prework_env, and 41 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const attn256RandBf16Scaled = @import("../attention/attn256.zig").attn256RandBf16Scaled;
const sigmoidTableFor = @import("../mlp/fused_activations.zig").sigmoidTableFor;

// ── Packed GDN prework mixer (mlxfast-challenge qwen35_packed_gdn_prework port) ──
//
// ONE launch replacing the GDN prework chain at decode and verify widths
// (S 1..9) — conv-state concat + depthwise conv1d + SiLU + q/k/v split +
// reshapes + Q/K rmsNorm-and-scale + conv-state slice + the forget gate
// (exp(-exp(A_log)·softplus(a+dt_bias))) + beta (sigmoid(b)) — ~14
// dispatches per GDN layer per step. Six outputs: normed/scaled Q and K,
// activated V, the next 3-row conv state, g and beta.
// The gate rides the SAME formulas MLX's compiled gate closure emits
// (precise exp, utils.h log1p, GT add rounding), so it is bit-identical
// to `computeGdnGate`; the in-kernel sigmoid (SiLU half, beta) is MLX's own
// unary formula (exp-of-abs form) — the challenge swept all 65,280 finite
// bf16 inputs against it: bit-exact everywhere.
// q/k/v, b and a are read through (offset, row stride) pairs (a combined
// projection could hand them over without slice copies).
// S < 3 rows of the next conv state come from the OLD state (row r reads
// concat(conv_state, qkv)[r + S]); before this generalization the kernel
// was gated to S >= 3 and decode ran the composed chain. Bit-identity with
// the composed chain is the acceptance bar.
const GDN_KERNEL_HEADER = kernelSource("kernels/gdn/log1p.metal");

const GDN_PREWORK_SOURCE = kernelSource("kernels/gdn/gdn_prework.metal");

var gdn_prework_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gdn_prework_engaged: bool = false;
pub var gdn_prework_override: ?bool = null; // test seam

fn gdnPreworkEnabled() bool {
    return gdn_prework_override orelse true;
}

pub var gdn_decode_fused_override: ?bool = null; // test seam

/// The decode-width GDN fusions (prework at S < 3, in-kernel gate/beta, the
/// norm-gate epilogue). Off = the pre-fusion decode path, op for op.
pub fn gdnDecodeFusedEnabled() bool {
    return gdn_decode_fused_override orelse true;
}

pub var gdn_decode_recur_engaged: bool = false;
pub var gdn_verify_recur_engaged: bool = false;
pub var gdn_verify_tree_engaged: bool = false;
pub var attn_qkv_joined_engaged: bool = false;
pub var gdn_verify_fold_engaged: bool = false;


/// Chain verify rows: gdn_decode.recurSeqFold (recurrence + norm-gate + conv input in one dispatch); the test seam
/// false keeps recurSeq -> norm-gate -> concat, the arm a declined fold falls back to.
pub var gdn_verify_fold_override: ?bool = null;
pub fn gdnVerifyFoldEnabled() bool {
    return gdn_verify_fold_override orelse true;
}

/// S=1 decode on packs without a Hadamard rotation: gdn_decode.recur then the norm-gate.
pub fn gdnDecodeRecurEnabled() bool {
    return gdnDecodeFusedEnabled();
}

fn getGdnPreworkKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_prework_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" };
    const output_names = [_][*:0]const u8{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_gdn_prework2",
        in_vec,
        out_vec,
        GDN_PREWORK_SOURCE,
        GDN_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_prework_kernel = kernel;
    return kernel;
}

const GdnPreworkCfgKey = struct { hk: c_int, hv: c_int, seq: c_int, batch: c_int, c: c_int, qs: c_int, qo: c_int, bs: c_int, bo: c_int, as: c_int, ao: c_int, dt: mlx.mlx_dtype, gt: mlx.mlx_dtype, gate: bool };
var gdn_prework_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gdn_prework_cfg_key: GdnPreworkCfgKey = std.mem.zeroes(GdnPreworkCfgKey);

pub const GdnPrework = struct {
    q: mlx.mlx_array, // [B, S, Hk, 128] normed + scaled
    k: mlx.mlx_array, // [B, S, Hk, 128] normed + scaled
    v: mlx.mlx_array, // [B, S, Hv, 128] silu-activated
    conv_state: mlx.mlx_array, // [B, NKEEP, C] next conv state
    g: mlx.mlx_array, // [B, S, Hv] forget gate
    beta: mlx.mlx_array, // [B, S, Hv] sigmoid(b)
};
/// Rows (batch*seq) the decode-width GDN fusions serve.
pub const GDN_FUSED_MAX_ROWS: c_int = 16;

pub var gdn_prefill_fused_override: ?bool = null;
pub var gdn_prefill_engaged = false;

pub fn gdnPrefillFusedFor(seq: c_int, batch: c_int) bool {
    if (seq < 17 or batch < 1 or batch > 2 or seq > @divTrunc(@import("../hc/hc_prefill.zig").max_seq, batch)) return false;
    return gdn_prefill_fused_override orelse true;
}

/// Inputs of the packed prework. `qkv`/`b`/`a` are read at `(off, stride)`
/// per row, so they may all alias one folded in_proj output.
pub const GdnPreworkArgs = struct {
    qkv: mlx.mlx_array, // rows of [.. QOFF + C ..] bf16
    qkv_off: c_int,
    qkv_stride: c_int,
    b: mlx.mlx_array, // rows of [.. BOFF + Hv ..] bf16
    b_off: c_int,
    b_stride: c_int,
    a: mlx.mlx_array,
    a_off: c_int,
    a_stride: c_int,
    A_log: mlx.mlx_array, // [Hv]
    dt_bias: mlx.mlx_array, // [Hv] bf16
    conv_state: mlx.mlx_array, // [B, NKEEP, C] bf16
    conv_w: mlx.mlx_array, // [C, 4, 1] bf16
    q_scale: mlx.mlx_array, // 0-dim bf16 (1/dk)
    k_scale: mlx.mlx_array, // 0-dim bf16 (1/sqrt(dk))
    qk_eps: f32 = 1e-6, // `ModelConfig.gdnQkNormEps`
    hk: c_int,
    hv: c_int,
    dk: c_int,
    dv: c_int,
    seq: c_int,
    batch: c_int = 1,
    /// False: no per-head gate (`g` comes back unwritten; KDA builds a per-channel one).
    gate: bool = true,
    /// The dtype `g` is computed and stored in: the recurrent state's. Null: the activations'.
    g_dtype: ?mlx.mlx_dtype = null,
};

/// One fused dispatch for the GDN prework at S 1..9 x B batched slots
/// (batch*seq <= GDN_FUSED_MAX_ROWS; per-head gate archs only). Null →
/// caller keeps the composed chain. The caller owns all six outputs (and
/// installs `conv_state` into the SSM cache entry).
pub fn gdnPreworkFused(s: mlx.mlx_stream, in: GdnPreworkArgs) !?GdnPrework {
    if (!gdnPreworkEnabled() or !mlx.streamIsGpu(s)) return null;
    if (!gdnPrefillFusedFor(in.seq, in.batch) and (in.seq < 1 or in.seq > 9 or in.batch < 1 or in.batch * in.seq > GDN_FUSED_MAX_ROWS)) return null;
    if (in.seq < 3 and !gdnDecodeFusedEnabled()) return null;
    if (in.dk != 128 or in.dv != 128) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    inline for (.{ in.b, in.conv_state, in.conv_w, in.q_scale, in.k_scale }) |arr| {
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    }
    if (in.gate and (mlx.mlx_array_dtype(in.a) != dt or mlx.mlx_array_dtype(in.dt_bias) != dt)) return null;
    const wsh = mlx.getShape(in.conv_w);
    if (wsh.len < 2 or wsh[1] != 4) return null;
    const c_dim: c_int = in.hk * in.dk * 2 + in.hv * in.dv;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != in.batch or qsh[1] != in.seq or qsh[2] != in.qkv_stride or in.qkv_off + c_dim > in.qkv_stride) return null;
    const bsh = mlx.getShape(in.b);
    if (bsh.len != 3 or bsh[0] != in.batch or bsh[1] != in.seq or bsh[2] != in.b_stride or in.b_off + in.hv > in.b_stride) return null;
    const ash = mlx.getShape(in.a);
    if (in.gate and (ash.len != 3 or ash[0] != in.batch or ash[1] != in.seq or ash[2] != in.a_stride or in.a_off + in.hv > in.a_stride)) return null;
    const csh = mlx.getShape(in.conv_state);
    if (csh.len != 3 or csh[0] != in.batch or csh[1] != 3 or csh[2] != c_dim) return null;

    const nkeep: c_int = 3;
    const gt = in.g_dtype orelse dt;
    const key = GdnPreworkCfgKey{ .hk = in.hk, .hv = in.hv, .seq = in.seq, .batch = in.batch, .c = c_dim, .qs = in.qkv_stride, .qo = in.qkv_off, .bs = in.b_stride, .bo = in.b_off, .as = in.a_stride, .ao = in.a_off, .dt = dt, .gt = gt, .gate = in.gate };
    if (gdn_prework_cfg == null or !std.meta.eql(gdn_prework_cfg_key, key)) {
        if (gdn_prework_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const q_shape = [_]c_int{ in.batch, in.seq, in.hk, in.dk };
        const v_shape = [_]c_int{ in.batch, in.seq, in.hv, in.dv };
        const st_shape = [_]c_int{ in.batch, nkeep, c_dim };
        const g_shape = [_]c_int{ in.batch, in.seq, in.hv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &v_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &st_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, gt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, in.batch * in.seq, 2 * in.hk + in.hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "GT", gt));
        const ints = .{ .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "NKEEP", nkeep }, .{ "C", c_dim }, .{ "TAB", @as(c_int, @intFromBool(in.seq >= 17)) }, .{ "QSTRIDE", in.qkv_stride }, .{ "QOFF", in.qkv_off }, .{ "BSTRIDE", in.b_stride }, .{ "BOFF", in.b_off }, .{ "ASTRIDE", in.a_stride }, .{ "AOFF", in.a_off }, .{ "GATE", @as(c_int, @intFromBool(in.gate)) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        gdn_prework_cfg = config;
        gdn_prework_cfg_key = key;
    }

    const kernel = try getGdnPreworkKernel();
    const seq_arr = mlx.mlx_array_new_int(in.seq);
    defer _ = mlx.mlx_array_free(seq_arr);
    const eps_arr = mlx.mlx_array_new_float(in.qk_eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const inputs_arr = [_]mlx.mlx_array{ in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.b, in.a, in.A_log, in.dt_bias, if (in.seq >= 17) try sigmoidTableFor(s, dt) else in.qkv, seq_arr, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gdn_prework_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 6) return error.MetalKernelBadOutputCount;
    var out = GdnPrework{
        .q = mlx.mlx_array_new(),
        .k = mlx.mlx_array_new(),
        .v = mlx.mlx_array_new(),
        .conv_state = mlx.mlx_array_new(),
        .g = mlx.mlx_array_new(),
        .beta = mlx.mlx_array_new(),
    };
    errdefer {
        inline for (.{ &out.q, &out.k, &out.v, &out.conv_state, &out.g, &out.beta }) |p| {
            if (p.*.ctx != null) _ = mlx.mlx_array_free(p.*);
        }
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.q, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.k, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.v, outputs_vec, 2));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, outputs_vec, 3));
    try mlx.check(mlx.mlx_vector_array_get(&out.g, outputs_vec, 4));
    try mlx.check(mlx.mlx_vector_array_get(&out.beta, outputs_vec, 5));
    if (!gdn_prework_engaged) {
        gdn_prework_engaged = true;
        log.info("[gdn] packed prework engaged: Hk={d} Hv={d} S={d} B={d}\n", .{ in.hk, in.hv, in.seq, in.batch });
    }
    return out;
}

// ── Fused GDN norm-gate epilogue ──
// rms_norm(y, norm_w) * silu(z) → flat [1, S, Hv*Dv] in ONE launch (the
// composed chain is rms_norm + sigmoid + multiply + multiply). Mirrors
// `rms_single_row`'s reduction (per-lane 4-element partials, simd_sum,
// precise rsqrt, `w * T(x * inv)`) and MLX's sigmoid, so it is bit-identical.
// z is read at (ZOFF, ZSTRIDE) so a wider projection row can hand it over.
const GDN_NORMGATE_SOURCE = kernelSource("kernels/gdn/gdn_normgate.metal");

var gdn_normgate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gdn_normgate_engaged: bool = false;
var gdn_normgate_declined: bool = false;
const GdnNormGateCfgKey = struct { hv: c_int, seq: c_int, batch: c_int, zs: c_int, zo: c_int, swish: bool, dt: mlx.mlx_dtype };
var gdn_normgate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gdn_normgate_cfg_key: GdnNormGateCfgKey = std.mem.zeroes(GdnNormGateCfgKey);

fn getGdnNormGateKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_normgate_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "y", "z", "norm_w", "eps", "sigtab" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("mlxserve_gdn_normgate", in_vec, out_vec, GDN_NORMGATE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_normgate_kernel = kernel;
    return kernel;
}

/// `rms_norm(y) * silu(z)` flattened to [B, S, Hv*Dv]. Null → composed chain.
pub fn gdnNormGateFused(
    s: mlx.mlx_stream,
    y: mlx.mlx_array, // [B, S, Hv, Dv] bf16
    z: mlx.mlx_array, // rows of [.. ZOFF + Hv*Dv ..] bf16
    z_off: c_int,
    z_stride: c_int,
    norm_w: mlx.mlx_array, // [Dv] bf16
    eps: mlx.mlx_array, // 0-dim f32
    swish: bool, // true: silu(z) * y (qwen3.5); false: y * sigmoid(z) (qwen4_exp, KDA)
    hv: c_int,
    dv: c_int,
    batch: c_int,
    seq: c_int,
) !?mlx.mlx_array {
    if (dv != 128 or !mlx.streamIsGpu(s)) return null;
    const prefill = gdnPrefillFusedFor(seq, batch);
    if (!prefill and !gdnDecodeFusedEnabled()) return null;
    if (!prefill and (seq < 1 or seq > GDN_FUSED_MAX_ROWS or batch < 1 or batch * seq > GDN_FUSED_MAX_ROWS)) return null;
    const dt = mlx.mlx_array_dtype(y);
    inline for (.{ y, z, norm_w }, 0..) |arr, i| {
        if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(arr) != dt) {
            if (!gdn_normgate_declined) {
                gdn_normgate_declined = true;
                log.info("[gdn] fused norm-gate declined: input {d} dtype {s}\n", .{ i, @tagName(mlx.mlx_array_dtype(arr)) });
            }
            return null;
        }
    }
    const ysh = mlx.getShape(y);
    const zsh = mlx.getShape(z);
    if (ysh.len != 4 or ysh[0] != batch or ysh[1] != seq or ysh[2] != hv or ysh[3] != dv or zsh.len != 3 or zsh[0] != batch or zsh[1] != seq or zsh[2] != z_stride or z_off + hv * dv > z_stride) {
        if (!gdn_normgate_declined) {
            gdn_normgate_declined = true;
            log.info("[gdn] fused norm-gate declined: y {any} z {any} z_off={d} z_stride={d} hv={d} dv={d} seq={d}\n", .{ ysh, zsh, z_off, z_stride, hv, dv, seq });
        }
        return null;
    }
    const key = GdnNormGateCfgKey{ .hv = hv, .seq = seq, .batch = batch, .zs = z_stride, .zo = z_off, .swish = swish, .dt = dt };
    if (gdn_normgate_cfg == null or !std.meta.eql(gdn_normgate_cfg_key, key)) {
        if (gdn_normgate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ batch, seq, hv * dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, batch * seq, hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        const ints = .{ .{ "HV", hv }, .{ "DV", dv }, .{ "ZSTRIDE", z_stride }, .{ "ZOFF", z_off }, .{ "TAB", @as(c_int, @intFromBool(seq >= 17)) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SWISH", @intFromBool(swish)));
        gdn_normgate_cfg = config;
        gdn_normgate_cfg_key = key;
    }
    const kernel = try getGdnNormGateKernel();
    const inputs_arr = [_]mlx.mlx_array{ y, z, norm_w, eps, if (seq >= 17) try sigmoidTableFor(s, dt) else y };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gdn_normgate_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gdn_normgate_engaged) {
        gdn_normgate_engaged = true;
        log.info("[gdn] fused norm-gate engaged: Hv={d} S={d} gate={s}\n", .{ hv, seq, if (swish) "swish" else "sigmoid" });
    }
    return out;
}

// ── tests ──

pub fn gdnParityRand(rnd: std.Random, shape: []const c_int, scale: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const b = try attn256RandBf16Scaled(rnd, shape, scale, s);
    defer _ = mlx.mlx_array_free(b);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, b, dt, s));
    return out;
}
