//! A pure op chain compiled into one closure (`mlx_compile`), built at load and applied per forward. Compiling is
//! not output-preserving, so a block whose legacy path ran a compiled chain carries the same closure.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// A compiled closure; `.ctx == null` = none (the caller runs the chain itself).
pub const Closure = mlx.mlx_closure;

/// `body(payload, inputs, outputs)` compiled with its own copy of `payload`; null when compiling fails or there is
/// no GPU backend (the callers then run the chain uncompiled, as the legacy path does).
pub fn compile(comptime P: type, payload: P, comptime n_out: usize, comptime body: fn (p: *const P, in: []const Array, out: *[n_out]Array) anyerror!void, shapeless: bool) Closure {
    if (mlx.noGpuBackend()) return .{ .ctx = null };
    const Thunk = struct {
        fn call(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
            const p: *const P = @ptrCast(@alignCast(raw.?));
            var ins: [8]Array = undefined;
            const n = mlx.mlx_vector_array_size(input);
            if (n > ins.len) return -1;
            for (0..n) |i| {
                ins[i] = mlx.mlx_array_new();
                if (mlx.mlx_vector_array_get(&ins[i], input, i) != 0) return -1;
            }
            defer for (ins[0..n]) |a| {
                _ = mlx.mlx_array_free(a);
            };
            var outs: [n_out]Array = undefined;
            body(p, ins[0..n], &outs) catch return -1;
            defer for (outs) |a| {
                _ = mlx.mlx_array_free(a);
            };
            res.* = mlx.mlx_vector_array_new_data(&outs, n_out);
            return 0;
        }
        fn free(raw: ?*anyopaque) callconv(.c) void {
            std.heap.c_allocator.destroy(@as(*P, @ptrCast(@alignCast(raw.?))));
        }
    };
    const p = std.heap.c_allocator.create(P) catch return .{ .ctx = null };
    p.* = payload;
    const raw = mlx.mlx_closure_new_func_payload(&Thunk.call, p, &Thunk.free);
    var out = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&out, raw, shapeless);
    _ = mlx.mlx_closure_free(raw);
    return if (rc == 0) out else .{ .ctx = null };
}

pub fn free(c: *Closure) void {
    if (c.ctx != null) _ = mlx.mlx_closure_free(c.*);
    c.* = .{ .ctx = null };
}

/// The closure's `n` outputs on `inputs`, owned by `g`; null when it returned another count.
pub fn apply(g: *Graph, c: Closure, inputs: []const Array, comptime n: usize) !?[n]Array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array{ .ctx = null };
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_closure_apply(&out_vec, c, in_vec));
    if (mlx.mlx_vector_array_size(out_vec) != n) return null;
    var out: [n]Array = undefined;
    for (&out, 0..) |*o, i| {
        var a = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&a, out_vec, i)) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        o.* = try g.adopt(a);
    }
    return out;
}
