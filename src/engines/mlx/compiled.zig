//! engines/mlx/compiled.
//! Owns compileGelu, compileGeglu, gegluClosureCallback, and 7 more.

const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const Transformer = @import("transformer.zig").Transformer;

/// Compile GELU activation for kernel fusion.
/// Fuses 8 separate ops into 1 GPU kernel, matching mlx-lm's @mx.compile behavior.
pub fn compileGelu(self: *Transformer) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &geluClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true); // shapeless=true
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_gelu = compiled;
        log.info("GELU compiled (kernel fusion enabled)\n", .{});
    } else {
        log.warn("GELU compilation failed, using uncompiled path\n", .{});
    }
}

/// Compile GeGLU: gelu(gate) * up → single fused kernel. The silu arm is
/// NOT routed here — mlx_compile's elementwise fusion does not reproduce
/// the op chain's intermediate rounding (a same-binary greedy A/B diverged
/// at token ~55), so silu gets the bit-exact `fusedSwiGLU` kernel instead.
pub fn compileGeglu(self: *Transformer) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &gegluClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_geglu = compiled;
        log.info("GeGLU compiled (kernel fusion enabled)\n", .{});
    }
}

pub fn gegluClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    var gate = mlx.mlx_array_new();
    var up = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&gate, input, 0) != 0) return -1;
    if (mlx.mlx_vector_array_get(&up, input, 1) != 0) {
        _ = mlx.mlx_array_free(gate);
        return -1;
    }
    defer _ = mlx.mlx_array_free(gate);
    defer _ = mlx.mlx_array_free(up);

    // geglu(gate, up) = gelu_approx(gate) * up
    const activated = self.geluUncompiled(gate) catch return -1;
    defer _ = mlx.mlx_array_free(activated);
    var result = mlx.mlx_array_new();
    mlx.check(mlx.mlx_multiply(&result, activated, up, self.s)) catch return -1;

    const out_arr = [_]mlx.mlx_array{result};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(result);
    return 0;
}

pub fn geluClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    var x = mlx.mlx_array_new();
    const get_rc = mlx.mlx_vector_array_get(&x, input, 0);
    if (get_rc != 0) return get_rc;
    defer _ = mlx.mlx_array_free(x);

    // gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    const result = self.geluUncompiled(x) catch return -1;

    const out_arr = [_]mlx.mlx_array{result};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(result);
    return 0;
}

pub fn closureIn(input: mlx.mlx_vector_array, i: usize) ?mlx.mlx_array {
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, i) != 0) return null;
    return a;
}

pub fn closureOut(res: *mlx.mlx_vector_array, a: mlx.mlx_array) c_int {
    const arr = [_]mlx.mlx_array{a};
    res.* = mlx.mlx_vector_array_new_data(&arr, 1);
    _ = mlx.mlx_array_free(a);
    return 0;
}

pub fn applyClosure(compiled: ?mlx.mlx_closure, inputs: []const mlx.mlx_array) !?mlx.mlx_array {
    const c = compiled orelse return null;
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array{ .ctx = null };
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_closure_apply(&out_vec, c, in_vec));
    if (mlx.mlx_vector_array_size(out_vec) != 1) return null;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, out_vec, 0));
    return out;
}

/// Free compiled JIT closures (compiled_gelu /
/// compiled_geglu / compiled_moe_routing). They get
/// bound to the calling thread's mlx GPU stream at compile time; once
/// inference moves to a different thread (Phase 2 scheduler) calls
/// against them fail with "no Stream(gpu, N) in current thread". Clear
/// them here so subsequent forward calls take the unfused fallback path,
/// then optionally re-warm on the new thread to recompile against its
/// own stream.
pub fn clearCompiledClosures(self: *Transformer) void {
    if (self.compiled_gelu) |c| {
        _ = mlx.mlx_closure_free(c);
        self.compiled_gelu = null;
    }
    if (self.compiled_geglu) |c| {
        _ = mlx.mlx_closure_free(c);
        self.compiled_geglu = null;
    }
    if (self.compiled_moe_routing) |c| {
        _ = mlx.mlx_closure_free(c);
        self.compiled_moe_routing = null;
    }
    if (self.compiled_hy3_routing) |c| {
        _ = mlx.mlx_closure_free(c);
        self.compiled_hy3_routing = null;
    }
    if (self.compiled_gdn_gate) |c| {
        _ = mlx.mlx_closure_free(c);
        self.compiled_gdn_gate = null;
    }
    inline for (.{ &self.compiled_hc_silu, &self.compiled_hc_mix, &self.compiled_hc_inj, &self.compiled_hc_write }) |slot| {
        if (slot.*) |c| {
            _ = mlx.mlx_closure_free(c);
            slot.* = null;
        }
    }
}
