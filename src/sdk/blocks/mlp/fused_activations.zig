//! sdk/blocks/mlp/fused_activations.
//! Owns SWIGLU_SOURCE, SWIGLU_CLAMP_SOURCE, swiglu_sig_tables, sigmoidTableFor, swigluSigTable, swiglu_engaged, and 27 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const ShapeKey = @import("../moe/router_kernel.zig").ShapeKey;

// ── Fused SwiGLU: silu(gate) * up in one dispatch ──
//
// `computeGeglu` only had a fused path for `gelu_approx`, so every SILU model —
// Laguna, Qwen, Llama, Mistral, i.e. most of the matrix — ran sigmoid +
// multiply + multiply as three dispatches, and a MoE layer pays it TWICE
// (routed experts + shared expert). It sits on the expert chain's critical
// path, so unlike the attention gate the launches are not overlapped: paired
// A/B on Laguna XS measured 13.12 -> 12.81 ms/forward, 3/3 pairs.
//
// Extending mlx_compile to the silu arm gets the same 2.4% but is NOT
// output-preserving — a same-binary greedy A/B diverged at token ~55, so the
// compiled elementwise fusion does not reproduce the chain's intermediate
// rounding. This kernel does: MLX's `Sigmoid` (the two-sided form) evaluated
// and ROUNDED TO T, then two `Multiply`s each promoting to float and rounding
// back, exactly as the three op kernels do.
// The sigmoid is a TABLE LOOKUP, not a recomputation, and that is the whole
// trick. Writing MLX's `Sigmoid` source verbatim is not enough to reproduce it:
// MLX's own kernels live in a metallib built with a different Metal math mode
// than the one `metal_kernel` JITs custom sources with (CompileOptions
// defaults to MathMode::Safe and mlx-c exposes no way to ask for another), so
// `1 / (1 + exp(|x|))` lands a rounding apart. An exhaustive sweep found it on
// exactly ONE of the 65536 bf16 patterns (-6.84375) — which random sampling
// missed and a 260-token greedy run hit within ~55 tokens, because a
// 8192-wide MLP draws 8192 values from that domain every layer.
//
// A 16-bit activation has only 65536 possible inputs, so `mlx_sigmoid` is
// evaluated over ALL of them once at first use and the kernel indexes the
// result by the input's bit pattern. Exact by construction, no transcendental
// in the hot loop, 128 KB resident. The two multiplies need no such care: a
// bf16/f16 product is exact in float, so one rounding is one rounding on any
// compiler.
pub const SWIGLU_SOURCE = kernelSource("kernels/mlp/swiglu.metal");

// SWIGLU_SOURCE behind the clamps of a `swiglu_limit` (GLM-5-Next): mlx Minimum on the gate,
// Clip on the up branch (exact in the tensor dtype), then the same chain.
pub const SWIGLU_CLAMP_SOURCE = kernelSource("kernels/mlp/swiglu_clamp.metal");

/// `mlx_sigmoid` tabulated over every 16-bit float pattern, indexed [CPU, GPU][bf16, f16]: the devices round apart.
var swiglu_sig_tables: [2][2]?mlx.mlx_array = @splat(@splat(null));

/// MLX's sigmoid of every 16-bit pattern of `dt` (bf16 or f16), for kernels
/// that must reproduce the op's rounding bit for bit.
pub fn sigmoidTableFor(s: mlx.mlx_stream, dt: mlx.mlx_dtype) !mlx.mlx_array {
    return swigluSigTable(s, dt, std.heap.c_allocator);
}

pub fn swigluSigTable(s: mlx.mlx_stream, dt: mlx.mlx_dtype, allocator: std.mem.Allocator) !mlx.mlx_array {
    const slot: usize = switch (dt) {
        .bfloat16 => 0,
        .float16 => 1,
        else => return error.UnsupportedSwigluDtype,
    };
    const cached = &swiglu_sig_tables[@intFromBool(mlx.streamIsGpu(s))][slot];
    if (cached.*) |t| return t;

    const vals = try all16BitPatterns(s, dt, allocator);
    defer _ = mlx.mlx_array_free(vals);
    var table = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(table);
    try mlx.check(mlx.mlx_sigmoid(&table, vals, s));
    // Materialize now: the table outlives every graph that reads it.
    try mlx.check(mlx.mlx_array_eval(table));
    cached.* = table;
    return table;
}

var swiglu_engaged: bool = false;
pub var swiglu_fused_override: ?bool = null;

pub fn swigluFusedEnabled() bool {
    return swiglu_fused_override orelse true;
}

/// `table[gate] * up`: an activation that is unary in x, tabulated over every
/// 16-bit pattern by running MLX's own op chain once, then one multiply.
pub const TABLE_MUL_SOURCE = kernelSource("kernels/mlp/table_mul.metal");

/// `x * c -> erf -> 1 + -> x * -> * half`, five ops in the activation dtype.
pub fn geluErfChain(x: mlx.mlx_array, one: mlx.mlx_array, half: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const inv_sqrt2 = try scalarOf(0.70710678118654752, mlx.mlx_array_dtype(x), s);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x, inv_sqrt2, s));
    var erf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(erf);
    try mlx.check(mlx.mlx_erf(&erf, scaled, s));
    var one_plus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one_plus);
    try mlx.check(mlx.mlx_add(&one_plus, one, erf, s));
    var x_times = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_times);
    try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, x_times, half, s));
    return result;
}

/// Every 16-bit pattern of `dt` as an array of that dtype (the domain a
/// per-dtype op table is evaluated over).
pub fn all16BitPatterns(s: mlx.mlx_stream, dt: mlx.mlx_dtype, allocator: std.mem.Allocator) !mlx.mlx_array {
    const N: usize = 1 << 16;
    const buf = try allocator.alloc(f32, N);
    defer allocator.free(buf);
    // Widened to the f32 that converts back EXACTLY, so astype is a relabel.
    for (buf, 0..) |*v, i| v.* = switch (dt) {
        .bfloat16 => @bitCast(@as(u32, @intCast(i)) << 16),
        else => @floatCast(@as(f16, @bitCast(@as(u16, @intCast(i))))),
    };
    const sh = [_]c_int{@intCast(N)};
    const v32 = mlx.mlx_array_new_data(buf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(v32);
    var vals = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&vals, v32, dt, s));
    return vals;
}

var gelu_erf_tables: [2]?mlx.mlx_array = @splat(null);

fn geluErfTable(s: mlx.mlx_stream, dt: mlx.mlx_dtype, one: mlx.mlx_array, half: mlx.mlx_array) !mlx.mlx_array {
    const slot: usize = if (dt == .bfloat16) 0 else 1;
    if (gelu_erf_tables[slot]) |t| return t;
    const vals = try all16BitPatterns(s, dt, std.heap.c_allocator);
    defer _ = mlx.mlx_array_free(vals);
    const table = try geluErfChain(vals, one, half, s);
    errdefer _ = mlx.mlx_array_free(table);
    if (mlx.mlx_array_dtype(table) != dt) return error.GeluTableDtype;
    try mlx.check(mlx.mlx_array_eval(table));
    gelu_erf_tables[slot] = table;
    return table;
}

const GateMulKind = enum(usize) {
    silu,
    gelu_erf,
    gelu_tanh,
    silu_clamp,
    fn name(self: GateMulKind) [*:0]const u8 {
        return switch (self) {
            .silu => "mlxserve_swiglu",
            .gelu_erf => "mlxserve_gelu_erf_mul",
            .gelu_tanh => "mlxserve_gelu_tanh_mul",
            .silu_clamp => "mlxserve_swiglu_clamp",
        };
    }
};
const gate_mul_kinds = @typeInfo(GateMulKind).@"enum".field_names.len;
var gate_mul_kernels: [gate_mul_kinds]?mlx.mlx_fast_metal_kernel = @splat(null);
const GateMulCfgKey = struct { shape: ShapeKey, dtype: mlx.mlx_dtype };
var gate_mul_cfg: [gate_mul_kinds]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var gate_mul_cfg_key: [gate_mul_kinds]GateMulCfgKey = @splat(std.mem.zeroes(GateMulCfgKey));
var gate_mul_engaged: [gate_mul_kinds]bool = @splat(false);

fn getGateMulKernel(kind: GateMulKind) !mlx.mlx_fast_metal_kernel {
    const k: usize = @backingInt(kind);
    if (gate_mul_kernels[k]) |kr| return kr;
    const input_names = [_][*:0]const u8{ "gate", "up", "sigtab", "N_size", "lim" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, if (kind == .silu_clamp) 5 else 4);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        kind.name(),
        in_vec,
        out_vec,
        switch (kind) {
            .silu => SWIGLU_SOURCE,
            .silu_clamp => SWIGLU_CLAMP_SOURCE,
            else => TABLE_MUL_SOURCE,
        },
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gate_mul_kernels[k] = kernel;
    return kernel;
}

/// `act(gate) * up`, elementwise, one dispatch, the activation read from a
/// per-dtype table of MLX's own results; `lim` (0-dim, the tensor dtype) for `.silu_clamp`.
/// Null → caller keeps the chain.
pub fn tableGateMul(s: mlx.mlx_stream, kind: GateMulKind, gate: mlx.mlx_array, up: mlx.mlx_array, table: mlx.mlx_array, lim: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != mlx.mlx_array_dtype(up)) return null;
    // FLOAT32 IS DECLINED ON PURPOSE. MLX ships its own kernels in a metallib
    // built with a different Metal math mode than the one `metal_kernel` JITs
    // custom sources with (CompileOptions defaults to MathMode::Safe, and
    // mlx-c exposes no way to ask for another), so an f32 `1 / (1 + exp(..))`
    // lands 1-2 ulps away from the stock sigmoid's. That is invisible in bf16
    // and f16 — both round the result far below the difference, and the test
    // pins them — but it is real in f32, and a fused path that is exact for
    // some dtypes and not others is worse than one that declines.
    if (dt != .bfloat16 and dt != .float16) return null;
    const gsh = mlx.getShape(gate);
    const ush = mlx.getShape(up);
    // No broadcasting: the gated MLP always pairs equal shapes, and a mismatch
    // here would silently read out of bounds rather than fail.
    if (gsh.len != ush.len or gsh.len == 0 or gsh.len > 5) return null;
    var n: i64 = 1;
    for (gsh, ush) |a, b| {
        if (a != b) return null;
        n *= a;
    }
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const k: usize = @backingInt(kind);

    const tg: c_int = @min(@as(c_int, 256), @max(@as(c_int, 32), @divTrunc(ni + 31, 32) * 32));
    const key = GateMulCfgKey{ .shape = ShapeKey.from(gsh), .dtype = dt };
    if (gate_mul_cfg[k] == null or !std.meta.eql(gate_mul_cfg_key[k], key)) {
        if (gate_mul_cfg[k]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, gsh.ptr, gsh.len, dt));
        // Grid is rounded UP to the threadgroup so the launch is uniform; the
        // kernel's own bound check drops the tail threads.
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        gate_mul_cfg[k] = config;
        gate_mul_cfg_key[k] = key;
    }

    const inputs_arr = [_]mlx.mlx_array{ gate, up, table, cachedScalarInt(ni), lim orelse table };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, if (kind == .silu_clamp) 5 else 4);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getGateMulKernel(kind);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gate_mul_cfg[k].?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gate_mul_engaged[k]) {
        gate_mul_engaged[k] = true;
        log.info("[mlp] fused {s} kernel engaged: n={d} tg={d}\n", .{ @tagName(kind), ni, tg });
    }
    return y;
}

/// `silu(gate) * up`, elementwise, one dispatch. Null → caller keeps the chain.
/// Exact by construction (bit-level sigmoid table), so callers outside this
/// file (tts.zig's talker/code-predictor MLPs) reuse it rather than growing a
/// second kernel.
// SWIGLU_SOURCE over a joined gate|up projection row (gate the first F
// columns): the two halves are read in place instead of copied out.
pub const SWIGLU_JOINED_SOURCE = kernelSource("kernels/mlp/swiglu_joined.metal");
var swiglu_joined_kernel: ?mlx.mlx_fast_metal_kernel = null;
var swiglu_joined_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var swiglu_joined_key: GateMulCfgKey = std.mem.zeroes(GateMulCfgKey);

/// `fusedSwiGLU(gate, up)` for `both` = [..., gate | up] (equal halves), bit-equal to it.
pub fn swigluJoined(s: mlx.mlx_stream, both: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(both);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sh = mlx.getShape(both);
    if (sh.len == 0 or sh.len > 5 or @rem(sh[sh.len - 1], 2) != 0) return null;
    var out_shape: [5]c_int = undefined;
    @memcpy(out_shape[0..sh.len], sh);
    const f = @divExact(sh[sh.len - 1], 2);
    out_shape[sh.len - 1] = f;
    var n: i64 = f;
    for (sh[0 .. sh.len - 1]) |d| n *= d;
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const key = GateMulCfgKey{ .shape = ShapeKey.from(sh), .dtype = dt };
    if (swiglu_joined_cfg == null or !std.meta.eql(swiglu_joined_key, key)) {
        if (swiglu_joined_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, sh.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + 255, 256) * 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "F", f));
        swiglu_joined_cfg = config;
        swiglu_joined_key = key;
    }
    if (swiglu_joined_kernel == null) {
        const in_names = [_][*:0]const u8{ "both", "sigtab", "N_size" };
        const out_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_swiglu_joined", in_vec, out_vec, SWIGLU_JOINED_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        swiglu_joined_kernel = k;
    }
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const ins = [_]mlx.mlx_array{ both, sigtab, cachedScalarInt(ni) };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, swiglu_joined_kernel.?, v, swiglu_joined_cfg.?, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

pub fn fusedSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    return tableGateMul(s, .silu, gate, up, sigtab, null);
}

/// `silu(min(gate, limit)) * clip(up, ±limit)` in one dispatch, bit for bit `clampedSwiGLU`.
pub fn fusedClampedSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, limit: f32) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const lim = try scalarOf(limit, dt, s);
    defer _ = mlx.mlx_array_free(lim);
    return tableGateMul(s, .silu_clamp, gate, up, sigtab, lim);
}

/// `gelu_tanh(x)` (whatever `Transformer.gelu` runs: compiled or the op
/// chain) tabulated per dtype. Null when the chain does not return `dt`.
pub var gelu_tanh_tables: [2]?mlx.mlx_array = @splat(null);
pub var gelu_tanh_declined: [2]bool = @splat(false);

/// `gelu_erf(gate) * up` in one dispatch, the activation tabulated from the
/// op chain itself (`geluErfTable`). Null → caller keeps the chain.
pub fn fusedGeluErfMul(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, one: mlx.mlx_array, half: mlx.mlx_array) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(one) != dt or mlx.mlx_array_dtype(half) != dt) return null;
    const table = try geluErfTable(s, dt, one, half);
    return tableGateMul(s, .gelu_erf, gate, up, table, null);
}
