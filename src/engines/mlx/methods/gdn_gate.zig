//! The legacy `Transformer`'s methods over `sdk/blocks/gdn/gate.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const log = @import("../../../sdk/util/log.zig");
const Transformer = @import("../transformer.zig").Transformer;
const gdnChainGateDtype = @import("../../../sdk/blocks/gdn/gate.zig").gdnChainGateDtype;
const gdnGateChain = @import("../../../sdk/blocks/gdn/gate.zig").gdnGateChain;

/// Compile the GatedDeltaNet gating chain (astype→exp→add→astype→exp→log1p→
/// multiply→negative→exp→astype, ~10 dispatches) into one fused kernel.
/// shapeless=true: pure elementwise chain, same trace for any [B,S,Hv].
pub fn compileGdnGate(self: *Transformer) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &gdnGateClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_gdn_gate = compiled;
        log.info("GDN gate compiled (kernel fusion enabled); recurrent state {s}, gate {s}\n", .{ @tagName(self.ssmStateDtype()), @tagName(gdn_decode.gateDtype(self.config.actDtype(), self.ssmStateDtype())) });
    }
}

/// Apply the compiled GDN gate closure if available, else the raw chain.
/// Returns owned g (the state's dtype, shape of `a`).
pub fn computeGdnGate(self: *const Transformer, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array) !mlx.mlx_array {
    if (self.compiled_gdn_gate) |compiled| {
        const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        if (mlx.mlx_vector_array_size(out_vec) == 1) {
            var g = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&g, out_vec, 0));
            return g;
        }
    }
    return gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s);
}

pub fn gdnGateClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    var A_log = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(A_log);
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
    defer _ = mlx.mlx_array_free(a);
    var dt_bias = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&dt_bias, input, 2) != 0) return -1;
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s) catch return -1;
    const out_arr = [_]mlx.mlx_array{g};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(g);
    return 0;
}
