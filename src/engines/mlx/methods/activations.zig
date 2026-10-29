//! The legacy `Transformer`'s methods over `sdk/blocks/mlp/activations.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const fused_activations = @import("../../../sdk/blocks/mlp/fused_activations.zig");
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const Transformer = @import("../transformer.zig").Transformer;
const all16BitPatterns = @import("../../../sdk/blocks/mlp/fused_activations.zig").all16BitPatterns;
const fusedClampedSwiGLU = @import("../../../sdk/blocks/mlp/fused_activations.zig").fusedClampedSwiGLU;
const fusedGeluErfMul = @import("../../../sdk/blocks/mlp/fused_activations.zig").fusedGeluErfMul;
const fusedSwiGLU = @import("../../../sdk/blocks/mlp/fused_activations.zig").fusedSwiGLU;
const geluErfChain = @import("../../../sdk/blocks/mlp/fused_activations.zig").geluErfChain;
const swigluFusedEnabled = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluFusedEnabled;
const tableGateMul = @import("../../../sdk/blocks/mlp/fused_activations.zig").tableGateMul;
const clampedSwiGLU = @import("../../../sdk/blocks/mlp/activations.zig").clampedSwiGLU;

/// GELU approximate: dispatches to compiled (fused kernel) when available.
pub fn gelu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    if (self.compiled_gelu) |compiled| {
        const in_arr = [_]mlx.mlx_array{x};
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
        return result;
    }
    return self.geluUncompiled(x);
}

/// Raw GELU implementation (8 ops, used as fallback and as compilation source).
pub fn geluUncompiled(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_power(&x3, x, self.three.?, self.s));
    var inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner);
    try mlx.check(mlx.mlx_multiply(&inner, self.gelu_inner.?, x3, self.s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));
    var scaled_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled_val);
    try mlx.check(mlx.mlx_multiply(&scaled_val, self.gelu_coeff.?, sum, self.s));
    var tanh_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tanh_val);
    try mlx.check(mlx.mlx_tanh(&tanh_val, scaled_val, self.s));
    var one_plus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one_plus);
    try mlx.check(mlx.mlx_add(&one_plus, self.one, tanh_val, self.s));
    var x_times = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_times);
    try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, self.s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, x_times, self.half, self.s));
    return result;
}

pub fn silu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, x, sig, self.s));
    return result;
}

pub inline fn mlpActivation(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    return switch (self.config.hidden_act) {
        .gelu_approx => self.gelu(x),
        .gelu => self.geluErf(x),
        .silu => self.silu(x),
        .relu_sq => self.reluSquared(x),
    };
}

/// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), in the activation dtype.
pub fn geluErf(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    return geluErfChain(x, self.one, self.half, self.s);
}

pub fn reluSquared(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    var relu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(relu);
    try mlx.check(mlx.mlx_maximum(&relu, x, zero, self.s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_square(&result, relu, self.s));
    return result;
}

pub fn geluTanhTable(self: *const Transformer, dt: mlx.mlx_dtype) !?mlx.mlx_array {
    const slot: usize = if (dt == .bfloat16) 0 else 1;
    if (fused_activations.gelu_tanh_tables[slot]) |t| return t;
    if (fused_activations.gelu_tanh_declined[slot]) return null;
    const vals = try all16BitPatterns(self.s, dt, std.heap.c_allocator);
    defer _ = mlx.mlx_array_free(vals);
    const table = try self.gelu(vals);
    if (mlx.mlx_array_dtype(table) != dt) {
        _ = mlx.mlx_array_free(table);
        fused_activations.gelu_tanh_declined[slot] = true;
        return null;
    }
    try mlx.check(mlx.mlx_array_eval(table));
    fused_activations.gelu_tanh_tables[slot] = table;
    return table;
}

/// `gelu_tanh(gate) * up` in one dispatch (tableGateMul). Null → chain.
pub fn fusedGeluTanhMul(self: *const Transformer, gate: mlx.mlx_array, up: mlx.mlx_array) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    const table = (try self.geluTanhTable(dt)) orelse return null;
    return tableGateMul(self.s, .gelu_tanh, gate, up, table, null);
}

/// act(gate) * up: one fused dispatch where an exact one exists, else the
/// compiled GeGLU closure, else the separate ops.
pub fn computeGeglu(self: *const Transformer, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    if (self.config.swiglu_clamp) return (try fusedClampedSwiGLU(self.s, gate, up, self.config.swiglu_limit)) orelse
        clampedSwiGLU(self.s, gate, up, self.config.swiglu_limit);
    if (self.config.hidden_act == .gelu_approx) {
        if (try self.fusedGeluTanhMul(gate, up)) |fused| return fused;
    }
    if (self.compiled_geglu) |compiled| {
        const in_arr = [_]mlx.mlx_array{ gate, up };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 2);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
        return result;
    }
    // silu / exact gelu: one bit-exact fused kernel (see fusedSwiGLU).
    if (self.config.hidden_act == .silu) {
        if (try fusedSwiGLU(self.s, gate, up)) |fused| return fused;
    } else if (self.config.hidden_act == .gelu) {
        if (try fusedGeluErfMul(self.s, gate, up, self.one, self.half)) |fused| return fused;
    }
    // Fallback: separate activation + multiply
    const activated = try self.mlpActivation(gate);
    defer _ = mlx.mlx_array_free(activated);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, activated, up, self.s));
    return result;
}

/// The clamped SwiGLU a `swiglu_limit` declares, dispatched instead of `computeGeglu`'s silu fast paths (their
/// sigmoid table is built for `silu(gate)*up` alone).
pub fn computeClampedSwiGLU(self: *const Transformer, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    return (try fusedClampedSwiGLU(self.s, gate, up, self.config.swiglu_limit)) orelse
        clampedSwiGLU(self.s, gate, up, self.config.swiglu_limit);
}

/// SwiGLU: silu(gate) * x
pub fn swiglu(self: *const Transformer, gate: mlx.mlx_array, x: mlx.mlx_array) !mlx.mlx_array {
    const activated = try self.silu(gate);
    defer _ = mlx.mlx_array_free(activated);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, activated, x, self.s));
    return result;
}
