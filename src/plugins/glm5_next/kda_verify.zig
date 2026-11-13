//! GLM-5.3-Flash's KDA verify rows that the joined decode step leaves to the op chain (a pack without the row-joined
//! projection, or gates it does not take): the bounded gate in one launch (exp(A_log) in it too), the capture's conv
//! input, and the per-row states as a launch MLX runs only when a rollback reads them. kda.zig runs them with the
//! packed prework and the norm-gate epilogue; each is the chain's bytes.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const log = @import("../../sdk/util/log.zig");
const decode_fused = @import("../../sdk/blocks/gdn/decode_fused.zig");
const kernels = @import("../../sdk/blocks/gdn/kernels.zig");
const kda = @import("kda.zig");

const Array = mlx.mlx_array;

/// The verify pieces that ran since the load (the tests read them).
pub const Runs = struct { prework: u64 = 0, gate: u64 = 0, normgate: u64 = 0, expsig: u64 = 0, lazy: u64 = 0 };
pub var runs: Runs = .{};

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn isFloat(d: mlx.mlx_dtype) bool {
    return d == .float16 or d == .bfloat16 or d == .float32;
}

/// MLX's promotion of two float dtypes: the dtype when both are it, else f32.
fn promote(a: mlx.mlx_dtype, b: mlx.mlx_dtype) ?mlx.mlx_dtype {
    if (!isFloat(a) or !isFloat(b)) return null;
    return if (a == b) a else .float32;
}

const GATE_SOURCE = kernelSource("plugins/glm5_next/kernels/kda_verify_gate.metal");
/// The gate's threadgroup: a row's channels are a whole number of them.
const gate_tg: c_int = 256;
var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var expsig_said = false;

fn gateKernel() !mlx.mlx_fast_metal_kernel {
    if (gate_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "a", "dt", "ea", "bound" };
    const outs = [_][*:0]const u8{"g"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("msv_glm_kda_verify_gate", iv, ov, GATE_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    gate_kernel = k;
    return k;
}

/// `kdaGateChain(a_log, a, dt_bias, bound)` over one slot's rows `a` [1, rows, n] in one launch: the f32 gate
/// [1, rows, n]. `in_exp`: exp(A_log) made in the launch, else by the chain's ops. Null off the GPU or on a shape or
/// dtype it does not take (the caller runs the chain).
pub fn gate(s: mlx.mlx_stream, a: Array, dt_bias: Array, a_log: Array, bound: f32, in_exp: bool) !?Array {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(a);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 1 or sh[2] < gate_tg or @rem(sh[2], gate_tg) != 0) return null;
    const n: usize = @intCast(sh[2]);
    if (mlx.mlx_array_size(dt_bias) != n or mlx.mlx_array_size(a_log) != n) return null;
    const add_t = promote(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(dt_bias)) orelse return null;
    if (!isFloat(mlx.mlx_array_dtype(a_log))) return null;
    // Without `in_exp`, the chain's exp(astype(A_log, f32)), made by its ops.
    var ea = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ea);
    if (in_exp) _ = mlx.mlx_array_set(&ea, a_log) else {
        var alf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(alf);
        try mlx.check(mlx.mlx_astype(&alf, a_log, .float32, s));
        try mlx.check(mlx.mlx_exp(&ea, alf, s));
    }
    const bound_arr = mlx.mlx_array_new_float(bound);
    defer _ = mlx.mlx_array_free(bound_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, sh[1], sh[2] }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, sh[2], sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, gate_tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "AddT", add_t));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", sh[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EXP", @intFromBool(in_exp)));
    const ins = [_]Array{ a, dt_bias, ea, bound_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try gateKernel(), iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 1) return error.MetalKernelBadOutputCount;
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_vector_array_get(&g, ov, 0));
    runs.gate += 1;
    if (in_exp) {
        runs.expsig += 1;
        if (!expsig_said) {
            expsig_said = true;
            log.info("[kda] verify gate exp(A_log) in its launch engaged: S={d} N={d}\n", .{ sh[1], sh[2] });
        }
    }
    return g;
}

/// conv1d.run's `input` for a verify, which a partial accept's rollback reads: [window | qkv] along the rows (MLX's
/// concatenate, as the chain makes it), a new array.
pub fn convInput(s: mlx.mlx_stream, window: Array, qkv: Array) !Array {
    const parts = [_]Array{ window, qkv };
    const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

pub const Lazy = struct { y: Array, state: Array, state_seq: Array };
var lazy_said = false;

/// kda.zig's capture branch as two launches over its inputs (q, k, v, the gate, beta, the state, T): y and the final
/// state from the stock recurrence (the capture text without its per-row stores, so the same bits), and the per-row
/// states [T, B, Hv, Dv, Dk] from the capture launch, kept alone: MLX runs it only when a rollback reads them.
pub fn captureLazy(s: mlx.mlx_stream, ins: [7]Array, in_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, hk: c_int, hv: c_int, dk: c_int, dv: c_int, batch: c_int, rows: c_int) !Lazy {
    const st_dt = mlx.mlx_array_dtype(ins[5]);
    const y_shape = [_]c_int{ batch, rows, hv, dv };
    const st_shape = [_]c_int{ batch, hv, dv, dk };
    var out: Lazy = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new(), .state_seq = mlx.mlx_array_new() };
    errdefer inline for (.{ out.y, out.state, out.state_seq }) |a| free(a);
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 4, out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &st_shape, 4, st_dt));
        try kernels.gdnSetGrid(cfg, .stock, dv, hv, batch);
        try kda.templateArgs(cfg, in_dt, st_dt, out_dt, dk, dv, hk, hv);
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernels.gdnRouteKernel(.stock, 0, true), iv, cfg, s));
        if (mlx.mlx_vector_array_size(ov) != 2) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&out.y, ov, 0));
        try mlx.check(mlx.mlx_vector_array_get(&out.state, ov, 1));
    }
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 4, out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, batch, hv, dv, dk }, 5, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &st_shape, 4, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, dv, batch * hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
        try kda.templateArgs(cfg, in_dt, st_dt, out_dt, dk, dv, hk, hv);
        const seq_stride = mlx.mlx_array_new_int(batch * hv * dv * dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const seq_ins = ins ++ [_]Array{seq_stride};
        const iv = mlx.mlx_vector_array_new_data(&seq_ins, seq_ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernels.getGdnKernelSeq(true), iv, cfg, s));
        if (mlx.mlx_vector_array_size(ov) != 3) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&out.state_seq, ov, 1));
    }
    runs.lazy += 1;
    if (!lazy_said) {
        lazy_said = true;
        log.info("[kda] verify lazy per-row states engaged: S={d} Hv={d}\n", .{ rows, hv });
    }
    return out;
}

// ── tests ──

const testing = std.testing;
const env = @import("../../env.zig");
const config = @import("config.zig");
const trunk = @import("trunk.zig");
const ops = @import("../../mlx/ops.zig");
const conv1d = @import("../../sdk/bases/blocks/conv1d.zig");
const kda_prologue = @import("kda_prologue.zig");
const kda_percore = @import("kda_percore.zig");
const kda_chunk = @import("kda_chunk.zig");
const kda_recurrence = @import("../../sdk/blocks/kda/recurrence.zig");

/// A GLM-5.3-Flash KDA layer at its served 64 heads of 128 (conv width 24576, gate width 8192), 4 conv taps, the
/// low-rank gates. At 4 heads a verify's sigmoids met none of the bf16 roundings the formula's fast exp moves.
const t_hidden: c_int = 512;
const t_heads: c_int = 64;
const t_head: c_int = 128;
const t_rank: c_int = 128;
const t_prompt: c_int = 17;
/// Verify widths: 1 + the drafted rows.
const verify_widths = [_]c_int{ 1, 2, 3, 4 };

const Switches = struct { verify: bool = false, expsig: bool = false, lazy: bool = false };

/// The switches as a run sets them, the process's own values put back by `restore`.
const Saved = struct {
    verify: env.Override(.MLX_SERVE_GLM_KDA_VERIFY),
    expsig: env.Override(.MLX_SERVE_GLM_KDA_EXPSIG),
    lazy: env.Override(.MLX_SERVE_GLM_KDA_LAZY_SEQ),

    fn init() Saved {
        return .{
            .verify = env.override(.MLX_SERVE_GLM_KDA_VERIFY, null),
            .expsig = env.override(.MLX_SERVE_GLM_KDA_EXPSIG, null),
            .lazy = env.override(.MLX_SERVE_GLM_KDA_LAZY_SEQ, null),
        };
    }

    fn restore(sv: *const Saved) void {
        sv.verify.restore();
        sv.expsig.restore();
        sv.lazy.restore();
    }
};

fn setSwitches(sw: Switches) void {
    env.set(.MLX_SERVE_GLM_KDA_VERIFY, if (sw.verify) "1" else "0");
    env.set(.MLX_SERVE_GLM_KDA_EXPSIG, if (sw.expsig) "1" else "0");
    env.set(.MLX_SERVE_GLM_KDA_LAZY_SEQ, if (sw.lazy) "1" else "0");
}

/// Seeded normal values of `dtype` at `scale`, evaluated.
fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dtype, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Seeded uniform values in [lo, hi) of `dtype`, evaluated.
fn uniform(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, lo: f32, hi: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, u, dtype, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A_log per head [h] expanded per key channel [h * 128], as trunk.zig loads it.
fn expandHeads(s: mlx.mlx_stream, per_head: Array, h: c_int) !Array {
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, per_head, &[_]c_int{ h, 1 }, 2, s));
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_broadcast_to(&bc, col, &[_]c_int{ h, t_head }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, bc, &[_]c_int{h * t_head}, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, flat, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The layer and its model: plain bf16 projections of seeded weights, pre-transposed as the loader leaves them.
const Rig = struct {
    cfg: config.Config,
    m: trunk.Model = undefined,
    la: trunk.Kda = undefined,
    w: [12]Array = @splat(.{ .ctx = null }),
    seed: u64 = 20261112,

    fn init(r: *Rig, s: mlx.mlx_stream, model_type: []const u8) !void {
        r.* = .{ .cfg = .{ .model_type = model_type, .hidden_size = @intCast(t_hidden), .linear_num_key_heads = @intCast(t_heads), .linear_num_value_heads = @intCast(t_heads), .linear_key_head_dim = @intCast(t_head), .linear_value_head_dim = @intCast(t_head), .linear_conv_kernel_dim = 4, .kda_gate_lower_bound = -5.0 } };
        r.m = .{ .c = &r.cfg, .s = s, .caps = .{}, .embed = .{}, .head = .{}, .final_norm = .{} };
        errdefer r.deinit();
        const dim = t_heads * t_head;
        r.w[0] = try normal(s, &r.seed, &.{ t_hidden, 3 * dim }, .bfloat16, 1.0 / 16.0);
        r.w[1] = try normal(s, &r.seed, &.{ t_hidden, t_rank }, .bfloat16, 1.0 / 16.0);
        r.w[2] = try normal(s, &r.seed, &.{ t_rank, dim }, .bfloat16, 1.0 / 11.0);
        r.w[3] = try normal(s, &r.seed, &.{ t_hidden, t_rank }, .bfloat16, 1.0 / 16.0);
        r.w[4] = try normal(s, &r.seed, &.{ t_rank, dim }, .bfloat16, 1.0 / 11.0);
        r.w[5] = try normal(s, &r.seed, &.{ t_hidden, t_heads }, .bfloat16, 1.0 / 16.0);
        r.w[6] = try normal(s, &r.seed, &.{ dim, t_hidden }, .bfloat16, 1.0 / 22.0);
        r.w[7] = try normal(s, &r.seed, &.{ 3 * dim, 4, 1 }, .bfloat16, 0.5);
        // A_log per head and per key channel, dt_bias, f32 as the packs load them; the gated norm's weight bf16.
        r.w[8] = try uniform(s, &r.seed, &.{t_heads}, .float32, 0.0, 2.5);
        r.w[9] = try expandHeads(s, r.w[8], t_heads);
        r.w[10] = try normal(s, &r.seed, &.{dim}, .float32, 1.0);
        r.w[11] = try uniform(s, &r.seed, &.{t_head}, .bfloat16, 0.5, 1.5);
        r.la = .{
            .qkv = .{ .w = r.w[0] },
            .z = .{ .w = r.w[1] },
            .z2 = .{ .w = r.w[2] },
            .a = .{ .w = r.w[3] },
            .a2 = .{ .w = r.w[4] },
            .b = .{ .w = r.w[5] },
            .out = .{ .w = r.w[6] },
            .conv_w = r.w[7],
            .A_log = r.w[9],
            .dt_bias = r.w[10],
            .norm_w = r.w[11],
            .a_log_h = r.w[8],
            .dt_f32 = r.w[10],
        };
    }

    fn deinit(r: *Rig) void {
        for (r.w) |a| free(a);
        r.w = @splat(.{ .ctx = null });
        for ([_]Array{ r.m.kda.q_scale, r.m.kda.k_scale, r.m.kda.ones, r.m.kda.eps }) |a| free(a);
        r.m.kda = .{};
    }

    fn input(r: *Rig, batch: c_int, rows: c_int) !Array {
        return normal(r.m.s, &r.seed, &.{ batch, rows, t_hidden }, .bfloat16, 1.0);
    }
};

fn newRef(a: Array) Array {
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, a);
    return out;
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// One verify's output, conv window, state, conv input and per-row states.
const Out = [5]Array;

fn freeOuts(outs: []Out) void {
    for (outs) |*o| for (o) |*a| {
        free(a.*);
        a.* = .{ .ctx = null };
    };
}

/// A cold 17-row prompt chunk, then a verify of `rows` rows for each of `xs` (each from the window and state the one
/// before it left), every one with capture under `sw`: what each verify leaves, evaluated. With LAZY_SEQ set, `lazy`
/// says whether its arm takes the call: the per-row states stay unrun after a full accept's reads, else the capture
/// launch has run them.
fn verifies(r: *Rig, x0: Array, xs: []const Array, rows: c_int, sw: Switches, lazy: bool, out: []Out) !void {
    setSwitches(sw);
    var e: kda.Entry = .{};
    defer e.deinit();
    const y0 = try kda.forward(&r.m, x0, &r.la, &e, 1, t_prompt, true, false);
    defer free(y0);
    try evalAll(&.{ y0, e.conv_state, e.ssm_state });
    for (xs, out) |x, *o| {
        // A verify of more than one row is a prefill-mode forward, a one-row verify a decode one.
        o[0] = try kda.forward(&r.m, x, &r.la, &e, 1, rows, rows > 1, true);
        o[1] = newRef(e.conv_state);
        o[2] = newRef(e.ssm_state);
        o[3] = newRef(e.spec_conv_input);
        o[4] = newRef(e.spec_state_seq);
        if (sw.lazy) {
            // Read as a full accept reads it: unrun where the arm takes the call, run by the capture launch where not.
            try evalAll(o[0..4]);
            var avail = true;
            try mlx.check(mlx._mlx_array_is_available(&avail, o[4]));
            try testing.expectEqual(!lazy, avail);
        }
        try evalAll(o);
    }
}

/// A cold prompt chunk, a verify of `xa` with its output, window, state and conv input read (not its per-row states),
/// the rollback of a partial accept of `acc` drafts (per-row state `acc` as the layer's state, the window kept), then
/// a verify of `xb` from it: the rolled state and the second verify's output, state and per-row states, evaluated.
fn rollback(r: *Rig, x0: Array, xa: Array, xb: Array, rows: c_int, acc: c_int, sw: Switches) ![4]Array {
    const s = r.m.s;
    setSwitches(sw);
    var e: kda.Entry = .{};
    defer e.deinit();
    const y0 = try kda.forward(&r.m, x0, &r.la, &e, 1, t_prompt, true, false);
    defer free(y0);
    try evalAll(&.{ y0, e.conv_state, e.ssm_state });
    const y1 = try kda.forward(&r.m, xa, &r.la, &e, 1, rows, rows > 1, true);
    defer free(y1);
    try evalAll(&.{ y1, e.conv_state, e.ssm_state, e.spec_conv_input });
    const sh = mlx.getShape(e.spec_state_seq);
    try testing.expect(sh.len == 5 and acc + 1 < sh[0]);
    var out: [4]Array = @splat(.{ .ctx = null });
    errdefer for (out) |a| free(a);
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_slice(&row, e.spec_state_seq, &[_]c_int{ acc, 0, 0, 0, 0 }, 5, &[_]c_int{ acc + 1, sh[1], sh[2], sh[3], sh[4] }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
    out[0] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out[0], row, &[_]c_int{ sh[1], sh[2], sh[3], sh[4] }, 4, s));
    free(e.ssm_state);
    e.ssm_state = newRef(out[0]);
    out[1] = try kda.forward(&r.m, xb, &r.la, &e, 1, rows, rows > 1, true);
    out[2] = newRef(e.ssm_state);
    out[3] = newRef(e.spec_state_seq);
    try evalAll(&out);
    return out;
}

fn bytesOf(a: Array) ![]const u8 {
    const n = mlx.mlx_array_size(a);
    return switch (mlx.mlx_array_dtype(a)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(a) orelse return error.TestNoData)[0..n]),
        .float16 => std.mem.sliceAsBytes((mlx.mlx_array_data_float16(a) orelse return error.TestNoData)[0..n]),
        .bfloat16 => std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(a) orelse return error.TestNoData)[0..n]),
        else => error.TestDtype,
    };
}

/// `got` holds `want`'s bits; with `lead`, over their first `lead` rows only.
fn expectBits(s: mlx.mlx_stream, want: Array, got: Array, lead: ?c_int) !void {
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| free(a);
    for ([_]Array{ want, got }, &parts) |a, *p| {
        var src = newRef(a);
        defer free(src);
        if (lead) |n| {
            const sh = mlx.getShape(a);
            const start: [8]c_int = @splat(0);
            var stop: [8]c_int = undefined;
            const stride: [8]c_int = @splat(1);
            for (sh, 0..) |d, i| stop[i] = d;
            stop[0] = n;
            var cut = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(cut);
            try mlx.check(mlx.mlx_slice(&cut, a, &start, sh.len, &stop, sh.len, &stride, sh.len, s));
            free(src);
            src = cut;
        }
        p.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(p, src, false, s));
    }
    try evalAll(&parts);
    const wb = try bytesOf(parts[0]);
    const gb = try bytesOf(parts[1]);
    if (std.mem.findDiff(u8, wb, gb)) |at| {
        std.debug.print("kda verify: bits differ from byte {d} of {d}\n", .{ at, wb.len });
        return error.TestBitsDiffer;
    }
}

/// Every output of `got` holds `want`'s bits; of the per-row states the T - 1 rows the capture kernel writes (its
/// last row is the final state's, unwritten by contract; a one-row verify writes none).
fn expectOuts(s: mlx.mlx_stream, want: []const Out, got: []const Out, rows: c_int) !void {
    for (want, got) |w, g| {
        for (w[0..4], g[0..4]) |a, b| try expectBits(s, a, b, null);
        if (rows > 1) try expectBits(s, w[4], g[4], rows - 1);
    }
}

fn runsSince(before: Runs) Runs {
    return .{ .prework = runs.prework - before.prework, .gate = runs.gate - before.gate, .normgate = runs.normgate - before.normgate, .expsig = runs.expsig - before.expsig, .lazy = runs.lazy - before.lazy };
}

/// The verify sets held to the chain: the texts alone, with exp(A_log) in the gate's launch, the lazy per-row states
/// alone, and everything.
const verify_sets = [_]Switches{
    .{ .verify = true },
    .{ .verify = true, .expsig = true },
    .{ .lazy = true },
    .{ .verify = true, .expsig = true, .lazy = true },
};

test "glm5_next kda verify: the prework, gate and norm-gate give the chain's output, window, state, conv input and per-row states bit for bit at 1 to 4 rows" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved = Saved.init();
    defer saved.restore();
    var r: Rig = undefined;
    try r.init(s, "glm5_next");
    defer r.deinit();
    const x0 = try r.input(1, t_prompt);
    defer free(x0);
    // 12 rows: the prework declines past 9, the gate and norm-gate still serve.
    for (verify_widths ++ [_]c_int{12}) |rows| {
        const xs: [2]Array = .{ try r.input(1, rows), try r.input(1, rows) };
        defer for (xs) |x| free(x);
        var want: [2]Out = @splat(@splat(.{ .ctx = null }));
        defer freeOuts(&want);
        const before_chain = runs;
        try verifies(&r, x0, &xs, rows, .{}, false, &want);
        try testing.expectEqual(Runs{}, runsSince(before_chain));
        for (verify_sets) |sw| {
            var got: [2]Out = @splat(@splat(.{ .ctx = null }));
            defer freeOuts(&got);
            const before = runs;
            try verifies(&r, x0, &xs, rows, sw, sw.lazy, &got);
            const ran = runsSince(before);
            const texts: u64 = if (sw.verify) 2 else 0;
            try testing.expectEqual(if (rows <= 9) texts else 0, ran.prework);
            try testing.expectEqual(texts, ran.gate);
            try testing.expectEqual(texts, ran.normgate);
            try testing.expectEqual(@as(u64, if (sw.lazy) 2 else 0), ran.lazy);
            try testing.expectEqual(@as(u64, if (sw.expsig) 2 else 0), ran.expsig);
            try expectOuts(s, &want, &got, rows);
        }
    }
}

test "glm5_next kda verify: Ling's KDA layer and two slots keep the chain" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved = Saved.init();
    defer saved.restore();
    setSwitches(.{ .verify = true, .expsig = true, .lazy = true });
    var r: Rig = undefined;
    try r.init(s, "bailing_hybrid");
    defer r.deinit();
    const before = runs;
    {
        const x0 = try r.input(1, t_prompt);
        defer free(x0);
        const xs: [1]Array = .{try r.input(1, 2)};
        defer free(xs[0]);
        var out: [1]Out = @splat(@splat(.{ .ctx = null }));
        defer freeOuts(&out);
        // every switch on, Ling's layer: the chain's capture launch runs the per-row states with the output
        try verifies(&r, x0, &xs, 2, .{ .verify = true, .expsig = true, .lazy = true }, false, &out);
    }
    r.cfg.model_type = "glm5_next";
    {
        const x = try r.input(2, 2);
        defer free(x);
        var e: kda.Entry = .{};
        defer e.deinit();
        const y = try kda.forward(&r.m, x, &r.la, &e, 2, 2, true, true);
        defer free(y);
        try evalAll(&.{ y, e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq });
    }
    try testing.expectEqual(Runs{}, runsSince(before));
}

test "glm5_next kda verify: after each partial accept's rollback the lazy per-row states and the texts give the chain's bits" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved = Saved.init();
    defer saved.restore();
    var r: Rig = undefined;
    try r.init(s, "glm5_next");
    defer r.deinit();
    const x0 = try r.input(1, t_prompt);
    defer free(x0);
    for (verify_widths) |rows| {
        const xa = try r.input(1, rows);
        defer free(xa);
        const xb = try r.input(1, rows);
        defer free(xb);
        var acc: c_int = 0;
        while (acc + 1 < rows) : (acc += 1) {
            const want = try rollback(&r, x0, xa, xb, rows, acc, .{});
            defer for (want) |a| free(a);
            for ([_]Switches{ .{ .lazy = true }, .{ .verify = true, .expsig = true, .lazy = true } }) |sw| {
                const got = try rollback(&r, x0, xa, xb, rows, acc, sw);
                defer for (got) |a| free(a);
                for (want[0..3], got[0..3]) |w, g| try expectBits(s, w, g, null);
                try expectBits(s, want[3], got[3], rows - 1);
            }
        }
    }
}

/// `n` values: the finite bf16 patterns (65,280: exponent not all ones) in order, from `from`, cycling.
fn finiteBf16(allocator: std.mem.Allocator, n: usize, from: usize) ![]f32 {
    const buf = try allocator.alloc(f32, n);
    var p: u32 = 0;
    var k: usize = 0;
    var seen: usize = 0;
    while (k < n) : (p = (p + 1) & 0xffff) {
        if ((p >> 7) & 0xff == 0xff) continue;
        if (seen >= from) {
            buf[k] = @bitCast(p << 16);
            k += 1;
        }
        seen += 1;
    }
    return buf;
}

/// `vals` as a bf16 array of `shape` (each an exact bf16, so the cast relabels).
fn bf16Of(s: mlx.mlx_stream, vals: []const f32, shape: []const c_int) !Array {
    const v32 = mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(v32);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, v32, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Elements of `want` and `got` (same shape and dtype, both contiguous after the copy) whose bits differ.
fn bitsApart(s: mlx.mlx_stream, want: Array, got: Array) !usize {
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| free(a);
    for ([_]Array{ want, got }, &parts) |a, *p| {
        p.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(p, a, false, s));
    }
    try evalAll(&parts);
    const wb = try bytesOf(parts[0]);
    const gb = try bytesOf(parts[1]);
    try testing.expectEqual(wb.len, gb.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < wb.len) : (i += 2) {
        if (!std.mem.eql(u8, wb[i .. i + 2], gb[i .. i + 2])) n += 1;
    }
    return n;
}

// Bar: at the served shape (64 heads, conv width 24576, 3 rows) a verify's launches give MLX's sigmoid op's bytes at
// every finite bf16 pattern (the conv's SiLU, beta, the norm-gate's sigmoid(z)): the chain's. The planted control is
// the formula arm the verify ran before (its bare `metal::exp` the fast exp in a runtime-compiled kernel): it must
// miss some, or the sweep sees nothing.
test "glm5_next kda verify: at the served shape the prework and the norm-gate give MLX's sigmoid at every finite bf16 pattern; the formula arm misses some" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = testing.allocator;
    const hk: c_int = 64;
    const hv: c_int = 64;
    const d: c_int = 128;
    const rows: c_int = 3;
    const c_dim: c_int = (2 * hk + hv) * d;
    const n_finite: usize = 65_280;
    var seed: u64 = 20261113;
    // The conv's newest tap alone, the window zeros: each conv output is its qkv value, so the 73,728 SiLU inputs
    // are every finite pattern.
    const w_buf = try a.alloc(f32, @intCast(c_dim * 4));
    defer a.free(w_buf);
    for (w_buf, 0..) |*v, i| v.* = if (i % 4 == 3) 1.0 else 0.0;
    const conv_w = try bf16Of(s, w_buf, &.{ c_dim, 4, 1 });
    defer free(conv_w);
    const q_vals = try finiteBf16(a, @intCast(rows * c_dim), 0);
    defer a.free(q_vals);
    const qkv = try bf16Of(s, q_vals, &.{ 1, rows, c_dim });
    defer free(qkv);
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{ 1, 3, c_dim }, 3, .bfloat16, s));
    var qs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qs);
    var ks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ks);
    const qs32 = mlx.mlx_array_new_float(1.0 / 128.0);
    defer _ = mlx.mlx_array_free(qs32);
    const ks32 = mlx.mlx_array_new_float(@sqrt(1.0 / 128.0));
    defer _ = mlx.mlx_array_free(ks32);
    try mlx.check(mlx.mlx_astype(&qs, qs32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&ks, ks32, .bfloat16, s));
    const eps: f32 = 1e-6 / 128.0;

    // The chain kda.forward runs for these rows: conv1d.run with its SiLU, q/k rms_norm over ones then the scale.
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const conv = try conv1d.run(&g, qkv, zero, conv_w, null, 1, c_dim, 4, true);
    const kd = hk * d;
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{d}, 1, .bfloat16, s));
    var ref: [3]Array = @splat(.{ .ctx = null });
    defer for (ref) |x| free(x);
    for ([_][2]c_int{ .{ 0, kd }, .{ kd, 2 * kd } }, 0..) |span, i| {
        const part = try g.reshape(try g.slice(conv.y, &.{ 0, 0, span[0] }, &.{ 1, rows, span[1] }), &.{ 1, rows, hk, d });
        var nrm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nrm);
        try mlx.check(mlx.mlx_fast_rms_norm(&nrm, part, ones, eps, s));
        ref[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&ref[i], nrm, if (i == 0) qs else ks, s));
    }
    ref[2] = g.escape(try g.reshape(try g.slice(conv.y, &.{ 0, 0, 2 * kd }, &.{ 1, rows, c_dim }), &.{ 1, rows, hv, d }));

    // beta: every finite pattern of b, 192 a launch (340 launches), against mlx_sigmoid; q, k and v at the first.
    var missed: [2]usize = .{ 0, 0 };
    var from: usize = 0;
    var first = true;
    while (from < n_finite) : (from += @intCast(rows * hv)) {
        const b_vals = try finiteBf16(a, @intCast(rows * hv), from);
        defer a.free(b_vals);
        const b = try bf16Of(s, b_vals, &.{ 1, rows, hv });
        defer free(b);
        var beta_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta_ref);
        try mlx.check(mlx.mlx_sigmoid(&beta_ref, b, s));
        const args: decode_fused.GdnPreworkArgs = .{ .qkv = qkv, .qkv_off = 0, .qkv_stride = c_dim, .b = b, .b_off = 0, .b_stride = hv, .a = b, .a_off = 0, .a_stride = hv, .A_log = b, .dt_bias = b, .conv_state = zero, .conv_w = conv_w, .q_scale = qs, .k_scale = ks, .qk_eps = eps, .hk = hk, .hv = hv, .dk = d, .dv = d, .seq = rows, .batch = 1, .gate = false, .table = true };
        var formula_args = args;
        formula_args.table = false;
        // the verify's prework
        const p = (try decode_fused.gdnPreworkFused(s, args)) orelse return error.TestPreworkDeclined;
        defer inline for (.{ p.q, p.k, p.v, p.conv_state, p.g, p.beta }) |x| free(x);
        try testing.expectEqual(@as(usize, 0), try bitsApart(s, beta_ref, p.beta));
        if (first) for ([_]Array{ p.q, p.k, p.v }, ref) |x, want| try testing.expectEqual(@as(usize, 0), try bitsApart(s, want, x));
        const f = (try decode_fused.gdnPreworkFused(s, formula_args)) orelse return error.TestPreworkDeclined;
        defer inline for (.{ f.q, f.k, f.v, f.conv_state, f.g, f.beta }) |x| free(x);
        missed[0] += try bitsApart(s, beta_ref, f.beta);
        if (first) missed[0] += try bitsApart(s, ref[2], f.v);
        first = false;
    }

    // The norm-gate: every finite pattern of z, 24,576 a launch, against rms_norm(y) * sigmoid(z).
    const norm_w = try uniform(s, &seed, &.{d}, .bfloat16, 0.5, 1.5);
    defer free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-5);
    defer _ = mlx.mlx_array_free(eps_arr);
    from = 0;
    while (from < n_finite) : (from += @intCast(rows * hv * d)) {
        const y = try normal(s, &seed, &.{ 1, rows, hv, d }, .bfloat16, 1.0);
        defer free(y);
        const z_vals = try finiteBf16(a, @intCast(rows * hv * d), from);
        defer a.free(z_vals);
        const z = try bf16Of(s, z_vals, &.{ 1, rows, hv * d });
        defer free(z);
        var nrm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nrm);
        try mlx.check(mlx.mlx_fast_rms_norm(&nrm, y, norm_w, 1e-5, s));
        var zh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zh);
        try mlx.check(mlx.mlx_reshape(&zh, z, &[_]c_int{ 1, rows, hv, d }, 4, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, zh, s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_multiply(&prod, nrm, sig, s));
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_reshape(&want, prod, &[_]c_int{ 1, rows, hv * d }, 3, s));
        const got = (try decode_fused.gdnNormGateChain(s, y, z, 0, hv * d, norm_w, eps_arr, false, hv, d, 1, rows)) orelse return error.TestNormGateDeclined;
        defer free(got);
        try testing.expectEqual(@as(usize, 0), try bitsApart(s, want, got));
        const formula = (try decode_fused.gdnNormGateFused(s, y, z, 0, hv * d, norm_w, eps_arr, false, hv, d, 1, rows)) orelse return error.TestNormGateDeclined;
        defer free(formula);
        missed[1] += try bitsApart(s, want, formula);
    }
    std.debug.print("kda verify sweep: the formula arm missed MLX's sigmoid at {d} prework and {d} norm-gate elements\n", .{ missed[0], missed[1] });
    try testing.expect(missed[0] + missed[1] > 0);
}

// Bar: a GLM layer's prompt with the prologue gives the chain's output, window and state bit for bit at the served
// heads: a cold 16,356-row chunk, then a warm 31-row chunk.
test "glm5_next kda prologue: a layer's prompt gives the chain's output, window and state, 16,356 rows cold then 31 warm" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const pro = env.override(.MLX_SERVE_KDA_PROMPT_PROLOGUE, null);
    defer pro.restore();
    var r: Rig = undefined;
    try r.init(s, "glm5_next");
    defer r.deinit();
    const xa = try r.input(1, 16_356);
    defer free(xa);
    const xb = try r.input(1, 31);
    defer free(xb);
    var outs: [2][6]Array = @splat(@splat(.{ .ctx = null }));
    defer for (&outs) |*o| for (o) |a| free(a);
    for ([_][*:0]const u8{ "0", "1" }, &outs) |on, *o| {
        env.set(.MLX_SERVE_KDA_PROMPT_PROLOGUE, on);
        const before = kda_prologue.runs;
        var e: kda.Entry = .{};
        defer e.deinit();
        o[0] = try kda.forward(&r.m, xa, &r.la, &e, 1, 16_356, true, false);
        o[1] = newRef(e.conv_state);
        o[2] = newRef(e.ssm_state);
        try evalAll(o[0..3]);
        o[3] = try kda.forward(&r.m, xb, &r.la, &e, 1, 31, true, false);
        o[4] = newRef(e.conv_state);
        o[5] = newRef(e.ssm_state);
        try evalAll(o[3..6]);
        try testing.expectEqual(if (on[0] == '1') before + 2 else before, kda_prologue.runs);
    }
    for (outs[0], outs[1]) |w, g| try expectBits(s, w, g, null);
}

/// Whether the two-a-core route takes the served heads here: it does where the stock per-core route declines (fewer
/// than 64 cores) and this GPU has a plan (32 cores or more) it launches.
fn percoreHere(s: mlx.mlx_stream) bool {
    if (kda_recurrence.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32)) return false;
    return kda_percore.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32);
}

/// A layer's prompt under the switches as set: a cold 16,356-row chunk, then a warm 31-row chunk, each chunk's output,
/// conv window and state, evaluated.
fn promptChunks(r: *Rig, xa: Array, xb: Array, o: *[6]Array) !void {
    var e: kda.Entry = .{};
    defer e.deinit();
    o[0] = try kda.forward(&r.m, xa, &r.la, &e, 1, 16_356, true, false);
    o[1] = newRef(e.conv_state);
    o[2] = newRef(e.ssm_state);
    try evalAll(o[0..3]);
    o[3] = try kda.forward(&r.m, xb, &r.la, &e, 1, 31, true, false);
    o[4] = newRef(e.conv_state);
    o[5] = newRef(e.ssm_state);
    try evalAll(o[3..6]);
}

/// ||got - want|| / ||want|| (L2, in f64) over two float arrays of one shape; a nonfinite `got` is an error.
fn normApart(s: mlx.mlx_stream, want: Array, got: Array) !f64 {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| free(a);
    for ([_]Array{ want, got }, &parts) |a, *p| {
        var wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&wide, a, .float32, s));
        p.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(p, wide, false, s));
    }
    try evalAll(&parts);
    const n = mlx.mlx_array_size(parts[0]);
    const w = mlx.mlx_array_data_float32(parts[0]) orelse return error.TestNoData;
    const g = mlx.mlx_array_data_float32(parts[1]) orelse return error.TestNoData;
    var num: f64 = 0;
    var den: f64 = 0;
    for (w[0..n], g[0..n]) |wv, gv| {
        if (!std.math.isFinite(gv)) return error.TestNonFinite;
        const d = @as(f64, gv) - @as(f64, wv);
        num += d * d;
        den += @as(f64, wv) * @as(f64, wv);
    }
    return @sqrt(num / den);
}

// Bar: with MLX_SERVE_KDA_PROMPT_PERCORE a layer's prompt (the other prompt switches at their defaults) keeps the stock
// path's conv window bit for bit and its output and state within the bound, the route run once a chunk.
test "glm5_next kda percore: rounding-class: a layer's prompt stays within 2^-5 of the stock prompt path, 16,356 rows cold then 31 warm" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!percoreHere(s)) return error.SkipZigTest;
    const pc = env.override(.MLX_SERVE_KDA_PROMPT_PERCORE, null);
    defer pc.restore();
    const pro = env.override(.MLX_SERVE_KDA_PROMPT_PROLOGUE, null);
    defer pro.restore();
    const ng = env.override(.MLX_SERVE_KDA_PROMPT_NORMGATE, null);
    defer ng.restore();
    var r: Rig = undefined;
    try r.init(s, "glm5_next");
    defer r.deinit();
    const xa = try r.input(1, 16_356);
    defer free(xa);
    const xb = try r.input(1, 31);
    defer free(xb);
    var outs: [2][6]Array = @splat(@splat(.{ .ctx = null }));
    defer for (&outs) |*o| for (o) |a| free(a);
    for ([_][*:0]const u8{ "0", "1" }, &outs) |on, *o| {
        env.set(.MLX_SERVE_KDA_PROMPT_PERCORE, on);
        const before = kda_percore.runs;
        try promptChunks(&r, xa, xb, o);
        try testing.expectEqual(if (on[0] == '1') before + 2 else before, kda_percore.runs);
    }
    for ([_]usize{ 1, 4 }) |i| try expectBits(s, outs[0][i], outs[1][i], null);
    // 2^-5: three bf16 roundings (y, the norm-gate, the out projection) each move an element by one ulp (2^-7) at most.
    const bound: f64 = 0x1p-5;
    for ([_]usize{ 0, 2, 3, 5 }) |i| {
        const apart = try normApart(s, outs[0][i], outs[1][i]);
        std.debug.print("kda percore: output {d} is {e} of the stock path's norm away from it\n", .{ i, apart });
        try testing.expect(apart <= bound);
    }
}

// Bar: the route gives the same bytes run after run, and whichever makes its q, k, v and beta (the prologue or the chain,
// each the chain's bytes) gives those bytes too.
test "glm5_next kda percore: deterministic: two runs give the same bytes, as does the prologue or the chain feeding it" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!percoreHere(s)) return error.SkipZigTest;
    const pc = env.override(.MLX_SERVE_KDA_PROMPT_PERCORE, "1");
    defer pc.restore();
    const pro = env.override(.MLX_SERVE_KDA_PROMPT_PROLOGUE, null);
    defer pro.restore();
    const ng = env.override(.MLX_SERVE_KDA_PROMPT_NORMGATE, null);
    defer ng.restore();
    var r: Rig = undefined;
    try r.init(s, "glm5_next");
    defer r.deinit();
    const xa = try r.input(1, 16_356);
    defer free(xa);
    const xb = try r.input(1, 31);
    defer free(xb);
    // PROLOGUE: the prologue twice, then the chain.
    const arms = [_][*:0]const u8{ "1", "1", "0" };
    var outs: [arms.len][6]Array = @splat(@splat(.{ .ctx = null }));
    defer for (&outs) |*o| for (o) |a| free(a);
    for (arms, &outs) |arm, *o| {
        env.set(.MLX_SERVE_KDA_PROMPT_PROLOGUE, arm);
        const before = kda_percore.runs;
        try promptChunks(&r, xa, xb, o);
        try testing.expectEqual(before + 2, kda_percore.runs);
    }
    for (outs[1..]) |o| for (outs[0], o) |w, g| try expectBits(s, w, g, null);
}

/// A cold prompt chunk `xa` of `rows` rows, then a warm 31-row chunk `xb`, through kda.forward: each one's output, conv
/// window and state, evaluated.
fn prompts(r: *Rig, xa: Array, rows: c_int, xb: Array) ![6]Array {
    var o: [6]Array = @splat(.{ .ctx = null });
    errdefer for (o) |a| free(a);
    var e: kda.Entry = .{};
    defer e.deinit();
    o[0] = try kda.forward(&r.m, xa, &r.la, &e, 1, rows, true, false);
    o[1] = newRef(e.conv_state);
    o[2] = newRef(e.ssm_state);
    try evalAll(o[0..3]);
    o[3] = try kda.forward(&r.m, xb, &r.la, &e, 1, 31, true, false);
    o[4] = newRef(e.conv_state);
    o[5] = newRef(e.ssm_state);
    try evalAll(o[3..6]);
    return o;
}

/// The largest |got - want| over the largest |want|, both in f32; dtypes and shapes alike.
fn relApart(s: mlx.mlx_stream, want: Array, got: Array) !f32 {
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var t: [7]Array = @splat(.{ .ctx = null });
    defer for (t) |a| free(a);
    for (&t) |*a| a.* = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&t[0], want, .float32, s));
    try mlx.check(mlx.mlx_astype(&t[1], got, .float32, s));
    try mlx.check(mlx.mlx_subtract(&t[2], t[1], t[0], s));
    try mlx.check(mlx.mlx_abs(&t[3], t[2], s));
    try mlx.check(mlx.mlx_max(&t[4], t[3], false, s));
    try mlx.check(mlx.mlx_abs(&t[5], t[0], s));
    try mlx.check(mlx.mlx_max(&t[6], t[5], false, s));
    try evalAll(&.{ t[4], t[6] });
    var apart: f32 = 0;
    var scale: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&apart, t[4]));
    try mlx.check(mlx.mlx_array_item_float32(&scale, t[6]));
    try testing.expect(scale > 0 and std.math.isFinite(apart));
    return apart / scale;
}

// Bound: bf16 outputs round at 2^-8 of their own magnitude and a summation order moves the f32 state far less, while a
// wrong gate, head or row moves them by their own size.
const rounding_bound: f32 = 1e-2;

/// The stock prompt route's two chunks (the chunkwise switches off) against `got`'s: the windows bit for bit (no lever
/// moves them), the outputs and states within `rounding_bound` of their largest magnitude.
fn expectNearStock(r: *Rig, xa: Array, rows: c_int, xb: Array, got: *const [6]Array) !void {
    const s = r.m.s;
    const ch = env.override(.MLX_SERVE_KDA_PROMPT_CHUNK, "0");
    defer ch.restore();
    const want = try prompts(r, xa, rows, xb);
    defer for (want) |a| free(a);
    for ([_]usize{ 1, 4 }) |i| try expectBits(s, want[i], got[i], null);
    for ([_]usize{ 0, 2, 3, 5 }) |i| {
        const apart = try relApart(s, want[i], got[i]);
        std.debug.print("kda prompt: output {d} apart from the stock route by {e} of its largest magnitude\n", .{ i, apart });
        try testing.expect(apart <= rounding_bound);
    }
}

/// Whether the copy's per-core route takes GLM's prompt rows on this GPU (the chunkwise levers then never run).
fn percoreTakes(s: mlx.mlx_stream) bool {
    return kda_recurrence.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32);
}

// Bar: the chunkwise route's output and state are the stock route's within the rounding bound and its window theirs bit
// for bit: a cold 16,356-row chunk, then a warm 31-row chunk.
test "glm5_next kda chunk: rounding-class: a layer's prompt on the chunkwise route meets the stock route, 16,356 rows cold then 31 warm" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (percoreTakes(s)) return error.SkipZigTest;
    if (!try kda_chunk.probe(s, t_heads, -5.0)) return error.SkipZigTest;
    const ch = env.override(.MLX_SERVE_KDA_PROMPT_CHUNK, "1");
    defer ch.restore();
    var r: Rig = undefined;
    try r.init(s, "glm5_next");
    defer r.deinit();
    const xa = try r.input(1, 16_356);
    defer free(xa);
    const xb = try r.input(1, 31);
    defer free(xb);
    const before = kda_chunk.runs;
    const got = try prompts(&r, xa, 16_356, xb);
    defer for (got) |a| free(a);
    try testing.expectEqual(before + 2, kda_chunk.runs);
    try expectNearStock(&r, xa, 16_356, xb, &got);
}

// Bar: Ling's layer keeps its route with the chunkwise switch on.
test "glm5_next kda chunk: Ling's layer never takes the chunkwise route" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ch = env.override(.MLX_SERVE_KDA_PROMPT_CHUNK, "1");
    defer ch.restore();
    var r: Rig = undefined;
    try r.init(s, "bailing_hybrid");
    defer r.deinit();
    const xa = try r.input(1, t_prompt);
    defer free(xa);
    const xb = try r.input(1, 31);
    defer free(xb);
    const before = kda_chunk.runs;
    const got = try prompts(&r, xa, t_prompt, xb);
    defer for (got) |a| free(a);
    try testing.expectEqual(before, kda_chunk.runs);
}
