//! GLM-5.3-Flash's KDA verify rows that the joined decode step leaves to the op chain (a pack without the row-joined
//! projection, or gates it does not take): the bounded gate in one launch and the capture's conv input. kda.zig runs
//! them with the packed prework and the norm-gate epilogue; each is the chain's bytes.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Array = mlx.mlx_array;

/// The verify pieces that ran since the load (the tests read them).
pub const Runs = struct { prework: u64 = 0, gate: u64 = 0, normgate: u64 = 0 };
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
/// [1, rows, n]. Null off the GPU or on a shape or dtype it does not take (the caller runs the chain).
pub fn gate(s: mlx.mlx_stream, a: Array, dt_bias: Array, a_log: Array, bound: f32) !?Array {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(a);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 1 or sh[2] < gate_tg or @rem(sh[2], gate_tg) != 0) return null;
    const n: usize = @intCast(sh[2]);
    if (mlx.mlx_array_size(dt_bias) != n or mlx.mlx_array_size(a_log) != n) return null;
    const add_t = promote(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(dt_bias)) orelse return null;
    if (!isFloat(mlx.mlx_array_dtype(a_log))) return null;
    // The chain's exp(astype(A_log, f32)), made by its ops.
    var alf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alf);
    try mlx.check(mlx.mlx_astype(&alf, a_log, .float32, s));
    var ea = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ea);
    try mlx.check(mlx.mlx_exp(&ea, alf, s));
    const bound_arr = mlx.mlx_array_new_float(bound);
    defer _ = mlx.mlx_array_free(bound_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, sh[1], sh[2] }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, sh[2], sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, gate_tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "AddT", add_t));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", sh[2]));
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

// ── tests ──

const testing = std.testing;
const env = @import("../../env.zig");
const config = @import("config.zig");
const trunk = @import("trunk.zig");
const kda = @import("kda.zig");

/// A GLM-5.3-Flash KDA layer at 4 heads of 128 (the texts take any head count), 4 conv taps, the low-rank gates.
const t_hidden: c_int = 512;
const t_heads: c_int = 4;
const t_head: c_int = 128;
const t_rank: c_int = 128;
const t_prompt: c_int = 17;
/// Verify widths: 1 + the drafted rows.
const verify_widths = [_]c_int{ 1, 2, 3, 4 };

const Switches = struct { verify: bool = false };

/// The switches as a run sets them, the process's own values put back by `restore`.
const Saved = struct {
    verify: env.Override(.MLX_SERVE_GLM_KDA_VERIFY),

    fn init() Saved {
        return .{ .verify = env.override(.MLX_SERVE_GLM_KDA_VERIFY, null) };
    }

    fn restore(sv: *const Saved) void {
        sv.verify.restore();
    }
};

fn setSwitches(sw: Switches) void {
    env.set(.MLX_SERVE_GLM_KDA_VERIFY, if (sw.verify) "1" else "0");
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
/// before it left), every one with capture under `sw`: what each verify leaves, evaluated.
fn verifies(r: *Rig, x0: Array, xs: []const Array, rows: c_int, sw: Switches, out: []Out) !void {
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
        try evalAll(o);
    }
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
    return .{ .prework = runs.prework - before.prework, .gate = runs.gate - before.gate, .normgate = runs.normgate - before.normgate };
}

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
        try verifies(&r, x0, &xs, rows, .{}, &want);
        try testing.expectEqual(Runs{}, runsSince(before_chain));
        var got: [2]Out = @splat(@splat(.{ .ctx = null }));
        defer freeOuts(&got);
        const before = runs;
        try verifies(&r, x0, &xs, rows, .{ .verify = true }, &got);
        const ran = runsSince(before);
        try testing.expectEqual(@as(u64, if (rows <= 9) 2 else 0), ran.prework);
        try testing.expectEqual(@as(u64, 2), ran.gate);
        try testing.expectEqual(@as(u64, 2), ran.normgate);
        try expectOuts(s, &want, &got, rows);
    }
}

test "glm5_next kda verify: Ling's KDA layer and two slots keep the chain" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved = Saved.init();
    defer saved.restore();
    setSwitches(.{ .verify = true });
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
        try verifies(&r, x0, &xs, 2, .{ .verify = true }, &out);
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
