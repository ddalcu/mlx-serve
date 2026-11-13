//! A KDA layer's exact verify in one launch (kernels/kda/kda_verify_rows.metal): row r has the bytes of the one-row
//! composed decode step on row r (conv1d + SiLU, the q/k rms norms and scales, kdaGateChain's bounded gate,
//! sigmoid(b), the stock vector-gate recurrence, rms_norm(y) * sigmoid(z)), the state stored to its dtype and loaded
//! back between rows as two decode forwards leave it. Serves one sequence, hk = hv, 128-wide heads, a 4-tap conv,
//! 16-bit activations; null elsewhere, checked before any launch.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const sigmoidTableFor = @import("../mlp/fused_activations.zig").sigmoidTableFor;

/// The most rows one launch takes: its threadgroup arrays hold this many.
pub const max_rows: c_int = 8;

pub const SOURCE: [:0]const u8 = kernelSource("kernels/kda/kda_verify_rows.metal");
pub const INPUTS = [_][*:0]const u8{ "qkv", "z", "a", "b", "conv_state", "conv_w", "A_log", "dt_bias", "state_in", "norm_w", "q_scale", "k_scale", "qk_eps", "norm_eps", "lb", "sigtab" };
pub const OUTPUTS = [_][*:0]const u8{ "y", "conv_out", "state_out", "state_seq" };

pub const In = struct {
    /// [1, S, C] q|k|v projections, C = 2 hk dk + hv dv
    qkv: mlx.mlx_array,
    /// [1, S, hv dv] output gate, [1, S, hv dk] gate input, [1, S, hv] beta input, in the activation dtype
    z: mlx.mlx_array,
    a: mlx.mlx_array,
    b: mlx.mlx_array,
    /// [1, 3, C] the conv window, [C, 4, 1] the depthwise conv
    conv_state: mlx.mlx_array,
    conv_w: mlx.mlx_array,
    /// f32 [hv dk] each (A_log expanded per key channel)
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    /// [1, hv, dv, dk] in the state dtype
    state: mlx.mlx_array,
    /// [dv]
    norm_w: mlx.mlx_array,
    /// 0-dim, the activation dtype: the scales the composed chain multiplies the normed q and k by
    q_scale: mlx.mlx_array,
    k_scale: mlx.mlx_array,
    qk_eps: f32,
    norm_eps: f32,
    /// The gate's lower bound (nonzero: the bounded sigmoid gate)
    bound: f32,
    hk: c_int,
    hv: c_int,
    dk: c_int,
    dv: c_int,
};

/// y [1, S, hv dv] (the out projection's input), the next conv window and state, the state after every row
/// [S, 1, hv, dv, dk] and the window's inputs [1, 3 + S, C]: what a verify's capture leaves for rollback.
pub const Out = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, state: mlx.mlx_array, state_seq: mlx.mlx_array, conv_input: mlx.mlx_array };

var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// Launches made: a served verify's engagement.
pub var launches: usize = 0;

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn isAct(dt: mlx.mlx_dtype) bool {
    return dt == .bfloat16 or dt == .float16;
}

fn shapeIs(a: mlx.mlx_array, want: []const c_int) bool {
    return std.mem.eql(c_int, mlx.getShape(a), want);
}

/// Whether one launch serves `in`. Past `max_rows` rows the kernel's threadgroup arrays would not hold them; a
/// softplus gate (bound 0) is another formula.
pub fn serves(s: mlx.mlx_stream, in: In) bool {
    if (!mlx.streamIsGpu(s) or in.bound == 0.0) return false;
    if (in.hk != in.hv or in.dk != 128 or in.dv != 128) return false;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != 1 or qsh[1] < 1 or qsh[1] > max_rows) return false;
    const rows = qsh[1];
    const c_dim = 2 * in.hk * in.dk + in.hv * in.dv;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (!isAct(dt) or qsh[2] != c_dim) return false;
    for ([_]mlx.mlx_array{ in.z, in.a, in.b, in.conv_state, in.conv_w, in.norm_w, in.q_scale, in.k_scale }) |x| {
        if (mlx.mlx_array_dtype(x) != dt) return false;
    }
    if (!shapeIs(in.z, &.{ 1, rows, in.hv * in.dv }) or !shapeIs(in.a, &.{ 1, rows, in.hv * in.dk }) or !shapeIs(in.b, &.{ 1, rows, in.hv })) return false;
    if (!shapeIs(in.conv_state, &.{ 1, 3, c_dim }) or !shapeIs(in.conv_w, &.{ c_dim, 4, 1 })) return false;
    if (mlx.mlx_array_size(in.norm_w) != @as(usize, @intCast(in.dv))) return false;
    const gate_n: usize = @intCast(in.hv * in.dk);
    for ([_]mlx.mlx_array{ in.A_log, in.dt_bias }) |x| {
        if (mlx.mlx_array_dtype(x) != .float32 or mlx.mlx_array_size(x) != gate_n) return false;
    }
    const st_dt = mlx.mlx_array_dtype(in.state);
    return shapeIs(in.state, &.{ 1, in.hv, in.dv, in.dk }) and (st_dt == .float32 or isAct(st_dt));
}

/// The verify's rows in one launch; null where `serves` declines.
pub fn verifyRows(s: mlx.mlx_stream, in: In) !?Out {
    if (!serves(s, in)) return null;
    const rows = mlx.getShape(in.qkv)[1];
    const c_dim = 2 * in.hk * in.dk + in.hv * in.dv;
    const dt = mlx.mlx_array_dtype(in.qkv);
    const st_dt = mlx.mlx_array_dtype(in.state);
    const k = rows_kernel orelse blk: {
        const iv = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const nk = mlx.mlx_fast_metal_kernel_new("msv_kda_verify_rows", iv, ov, SOURCE.ptr, "", true, false);
        if (nk.ctx == null) return error.MetalKernelCompileFailed;
        rows_kernel = nk;
        break :blk nk;
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, in.hv * in.dv }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, c_dim }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, in.hv, in.dv, in.dk }, 4, st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, in.hv, in.dv, in.dk }, 4, st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * in.hv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", st_dt));
    inline for (.{ .{ "TOK", rows }, .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "C", c_dim }, .{ "NKEEP", @as(c_int, 3) } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const qe = mlx.mlx_array_new_float(in.qk_eps);
    defer free(qe);
    const ne = mlx.mlx_array_new_float(in.norm_eps);
    defer free(ne);
    const lb = mlx.mlx_array_new_float(in.bound);
    defer free(lb);
    // MLX's own sigmoid of every 16-bit pattern (cached per dtype): each sigmoid of a T value reads it.
    const sigtab = try sigmoidTableFor(s, dt);
    const inputs = [_]mlx.mlx_array{ in.qkv, in.z, in.a, in.b, in.conv_state, in.conv_w, in.A_log, in.dt_bias, in.state, in.norm_w, in.q_scale, in.k_scale, qe, ne, lb, sigtab };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != OUTPUTS.len) return error.MetalKernelBadOutputCount;
    var out: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer for (out) |o| free(o);
    for (&out, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
    var seq = mlx.mlx_array_new();
    errdefer free(seq);
    try mlx.check(mlx.mlx_reshape(&seq, out[3], &[_]c_int{ rows, 1, in.hv, in.dv, in.dk }, 5, s));
    var conv_input = mlx.mlx_array_new();
    errdefer free(conv_input);
    {
        const parts = [_]mlx.mlx_array{ in.conv_state, in.qkv };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
    }
    free(out[3]);
    launches += 1;
    return .{ .y = out[0], .conv_state = out[1], .state = out[2], .state_seq = seq, .conv_input = conv_input };
}

// ── tests ──

const ops = @import("../../../mlx/ops.zig");
const conv1d = @import("../../bases/blocks/conv1d.zig");
const kdaGateChain = @import("gate.zig").kdaGateChain;
const gdn_kernels = @import("../gdn/kernels.zig");

const Graph = ops.Graph;
const Array = mlx.mlx_array;

/// A KDA layer's operands at `hv` heads of 128 (Ling's bounded gate at -5, a bf16 state), drawn from `seed`.
const Case = struct {
    hv: c_int,
    rows: c_int,
    qkv: Array = undefined,
    z: Array = undefined,
    a: Array = undefined,
    b: Array = undefined,
    conv_state: Array = undefined,
    conv_w: Array = undefined,
    A_log: Array = undefined,
    dt_bias: Array = undefined,
    state: Array = undefined,
    norm_w: Array = undefined,
    q_scale: Array = undefined,
    k_scale: Array = undefined,

    const D: c_int = 128;
    const bound: f32 = -5.0;
    const eps: f32 = 1e-6;

    fn cdim(c: *const Case) c_int {
        return 3 * c.hv * D;
    }

    /// normal(loc, scale) in `dt`, evaluated.
    fn normal(g: *Graph, seed: u64, sh: []const c_int, loc: f32, scale: f32, dt: mlx.mlx_dtype) !Array {
        const key = try g.call(mlx.mlx_random_key, .{seed});
        const f = try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, @as(mlx.mlx_dtype, .float32), loc, scale, key });
        const out = try g.astype(f, dt);
        try ops.eval(out);
        return out;
    }

    /// The operands; `views`: the window and the state as offset views (a rollback's slices) and the conv weight as
    /// the loader's swapped copy of an HF `[C, 1, 4]` tensor.
    fn draw(c: *Case, g: *Graph, seed: u64, views: bool) !void {
        const bf: mlx.mlx_dtype = .bfloat16;
        const C = c.cdim();
        const W = c.hv * D;
        c.qkv = try normal(g, seed, &.{ 1, c.rows, C }, 0, 1, bf);
        c.z = try normal(g, seed + 1, &.{ 1, c.rows, W }, 0, 6, bf);
        c.a = try normal(g, seed + 2, &.{ 1, c.rows, W }, 0, 6, bf);
        c.b = try normal(g, seed + 3, &.{ 1, c.rows, c.hv }, 0, 6, bf);
        c.dt_bias = try normal(g, seed + 4, &.{W}, 0, 0.5, .float32);
        c.norm_w = try normal(g, seed + 5, &.{D}, 1, 0.2, bf);
        // A_log per head (log of 1..16), expanded per key channel as the loader keeps it.
        const heads = [_]c_int{c.hv};
        const u = try g.call(mlx.mlx_random_uniform, .{ try g.scalar(1, .float32), try g.scalar(16, .float32), &heads, heads.len, @as(mlx.mlx_dtype, .float32), try g.call(mlx.mlx_random_key, .{seed + 6}) });
        c.A_log = try g.reshape(try g.broadcast(try g.reshape(try g.log(u), &.{ c.hv, 1 }), &.{ c.hv, D }), &.{W});
        try ops.eval(c.A_log);
        if (views) {
            const prev = try normal(g, seed + 7, &.{ 1, 5, C }, 0, 1, bf);
            c.conv_state = try g.slice(prev, &.{ 0, 2, 0 }, &.{ 1, 5, C });
            const seq = try normal(g, seed + 8, &.{ 3, 1, c.hv, D, D }, 0, 0.1, bf);
            c.state = try g.reshape(try g.slice(seq, &.{ 1, 0, 0, 0, 0 }, &.{ 2, 1, c.hv, D, D }), &.{ 1, c.hv, D, D });
            const raw = try normal(g, seed + 9, &.{ C, 1, 4 }, 0, 0.5, bf);
            c.conv_w = try g.add(try g.call(mlx.mlx_swapaxes, .{ raw, @as(c_int, 1), @as(c_int, 2) }), try g.zeros(&.{1}, bf));
        } else {
            c.conv_state = try normal(g, seed + 7, &.{ 1, 3, C }, 0, 1, bf);
            c.state = try normal(g, seed + 8, &.{ 1, c.hv, D, D }, 0, 0.1, bf);
            c.conv_w = try normal(g, seed + 9, &.{ C, 4, 1 }, 0, 0.5, bf);
        }
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(D));
        c.q_scale = try g.scalar(inv, bf);
        c.k_scale = try g.scalar(@sqrt(inv), bf);
        for ([_]Array{ c.conv_state, c.state, c.conv_w, c.q_scale, c.k_scale }) |x| try ops.eval(x);
    }

    fn inputs(c: *const Case) In {
        return .{ .qkv = c.qkv, .z = c.z, .a = c.a, .b = c.b, .conv_state = c.conv_state, .conv_w = c.conv_w, .A_log = c.A_log, .dt_bias = c.dt_bias, .state = c.state, .norm_w = c.norm_w, .q_scale = c.q_scale, .k_scale = c.k_scale, .qk_eps = eps, .norm_eps = eps, .bound = bound, .hk = c.hv, .hv = c.hv, .dk = D, .dv = D };
    }

    /// One serial decode step of the composed layer on row `r` from `window` and `state`: y [1, 1, hv dv], the next
    /// window and state.
    fn step(c: *const Case, g: *Graph, r: c_int, window: Array, state: Array) !struct { y: Array, window: Array, state: Array } {
        const C = c.cdim();
        const W = c.hv * D;
        const bf: mlx.mlx_dtype = .bfloat16;
        const conv = try conv1d.run(g, try g.sliceAxis(c.qkv, 1, r, r + 1), window, c.conv_w, null, 1, C, 4, true);
        const ones = try g.ones(&.{D}, bf);
        var qk: [2]Array = undefined;
        for (&qk, [_]Array{ c.q_scale, c.k_scale }, 0..) |*o, sc, p| {
            const off: c_int = @intCast(p);
            const heads = try g.reshape(try g.slice(conv.y, &.{ 0, 0, off * W }, &.{ 1, 1, (off + 1) * W }), &.{ 1, 1, c.hv, D });
            o.* = try g.mul(try g.rmsNorm(heads, ones, eps), sc);
        }
        const v = try g.reshape(try g.slice(conv.y, &.{ 0, 0, 2 * W }, &.{ 1, 1, C }), &.{ 1, 1, c.hv, D });
        const gate = try g.adopt(try kdaGateChain(c.A_log, try g.sliceAxis(c.a, 1, r, r + 1), c.dt_bias, bound, g.s));
        const beta = try g.sigmoid(try g.sliceAxis(c.b, 1, r, r + 1));
        // The stock vector-gate kernel, as the layer launches it at one row.
        const kcfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(kcfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ 1, 1, c.hv, D }, 4, bf));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ 1, c.hv, D, D }, 4, bf));
        try gdn_kernels.gdnSetGrid(kcfg, .stock, D, c.hv, 1);
        inline for (.{ "InT", "StT", "OutT" }) |n| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, n, bf));
        inline for (.{ .{ "Dk", D }, .{ "Dv", D }, .{ "Hk", c.hv }, .{ "Hv", c.hv } }) |ta| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(kcfg, ta[0], ta[1]));
        const t_one = try g.adopt(mlx.mlx_array_new_int(1));
        const ins = [_]Array{ qk[0], qk[1], v, gate, beta, state, t_one };
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try gdn_kernels.gdnRouteKernel(.stock, 0, true), iv, kcfg, g.s));
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
        _ = try g.adopt(y);
        var st = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&st, ov, 1));
        _ = try g.adopt(st);
        const z = try g.reshape(try g.sliceAxis(c.z, 1, r, r + 1), &.{ 1, 1, c.hv, D });
        const gated = try g.mul(try g.rmsNorm(y, c.norm_w, eps), try g.sigmoid(z));
        return .{ .y = try g.reshape(gated, &.{ 1, 1, W }), .window = conv.window, .state = st };
    }
};

/// The launch over `rows` rows against `rows` serial composed steps: y, the state after every row, the window's inputs,
/// the next window and the state, bit for bit.
fn expectSerial(hv: c_int, rows: c_int, seed: u64, views: bool) !void {
    var g = Graph.init(std.testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var c = Case{ .hv = hv, .rows = rows };
    try c.draw(&g, seed, views);
    var window = c.conv_state;
    var state = c.state;
    var ys: [8]Array = undefined;
    var states: [8]Array = undefined;
    const n: usize = @intCast(rows);
    for (0..n) |r| {
        const st = try c.step(&g, @intCast(r), window, state);
        ys[r] = st.y;
        states[r] = st.state;
        window = st.window;
        state = st.state;
    }
    const got = (try verifyRows(g.s, c.inputs())) orelse return error.TestDeclined;
    for ([_]Array{ got.y, got.conv_state, got.state, got.state_seq, got.conv_input }) |x| _ = try g.adopt(x);
    try ops.expectBitEqual(got.y, try g.concat(ys[0..n], 1));
    const sv = mlx.mlx_vector_array_new_data(&states, n);
    defer _ = mlx.mlx_vector_array_free(sv);
    try ops.expectBitEqual(got.state_seq, try g.call(mlx.mlx_stack_axis, .{ sv, @as(c_int, 0) }));
    try ops.expectBitEqual(got.conv_input, try g.concat(&.{ c.conv_state, c.qkv }, 1));
    try ops.expectBitEqual(got.conv_state, window);
    try ops.expectBitEqual(got.state, state);
}

// Bar: n rows in one launch are n serial one-row composed steps, bit for bit (the fixture's 4 heads and Ling's 64).
test "kda verify rows: n rows in one launch are n serial one-row decode steps, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var rows: c_int = 1;
    while (rows <= max_rows) : (rows += 1) try expectSerial(4, rows, 0x5E00 + 16 * @as(u64, @intCast(rows)), false);
    for ([_]c_int{ 2, 5, 8 }) |r| try expectSerial(64, r, 0x6E00 + @as(u64, @intCast(r)), false);
}

// Bar: the same on operands as served: an offset window and state (a rollback's views), the loader's swapped conv.
test "kda verify rows: views of a rollback's window and state give serial decode's bytes too" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    for ([_]c_int{ 2, 3, 8 }) |r| try expectSerial(4, r, 0x7E00 + @as(u64, @intCast(r)), true);
}

// Bar: past 8 rows, a softplus gate or 64-wide heads the launch declines before it runs.
test "kda verify rows: shapes it does not serve are declined before any launch" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(std.testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var c = Case{ .hv = 4, .rows = 9 };
    try c.draw(&g, 0x8E00, false);
    const before = launches;
    try std.testing.expect(try verifyRows(g.s, c.inputs()) == null);
    var soft = c;
    soft.rows = 2;
    try soft.draw(&g, 0x8E10, false);
    var in = soft.inputs();
    in.bound = 0;
    try std.testing.expect(try verifyRows(g.s, in) == null);
    in = soft.inputs();
    in.dk = 64;
    try std.testing.expect(try verifyRows(g.s, in) == null);
    try std.testing.expectEqual(before, launches);
}
