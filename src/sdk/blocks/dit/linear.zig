//! A media projection on the op layer: dense or affine-quantized, optional bias. The same ops
//! in the same order as `mlx/scope.zig`'s `Linear`, so a backend moves over bit for bit.
//! Handles are borrowed (from the backend's weight map).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;

pub const Linear = struct {
    w: Array,
    /// Present on an affine-quantized weight; `bits`/`group_size` solved at load.
    scales: ?Array = null,
    biases: Array = .{ .ctx = null },
    bias: ?Array = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    pub fn forward(self: *const Linear, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const y = if (self.scales) |sca|
            try g.qmm(x, self.w, sca, self.biases, @intCast(self.group_size), @intCast(self.bits))
        else
            try g.matmul(x, try g.transpose(self.w, &.{ 1, 0 }));
        return parent.keep(if (self.bias) |b| try g.add(y, b) else y);
    }
};

// ── tests ──

const testing = std.testing;
const kit = @import("testkit.zig");
const scope_mod = @import("../../../mlx/scope.zig");

test "media blocks: dit Linear: bit-equal to mlx/scope.zig's Linear (dense, biased, affine-quantized)" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const x = try kit.randn(&g, &.{ 1, 5, 64 }, 1, .bfloat16, 1);
    const w = try kit.randn(&g, &.{ 32, 64 }, 2, .bfloat16, 0.1);
    const b = try kit.randn(&g, &.{32}, 3, .bfloat16, 0.1);

    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var parts: [3]Array = undefined;
    for (&parts, 0..) |*p, i| {
        p.* = mlx.mlx_array_new();
        _ = mlx.mlx_vector_array_get(p, qv, i);
        _ = try g.adopt(p.*);
    }

    const cases = [_]Linear{
        .{ .w = w },
        .{ .w = w, .bias = b },
        .{ .w = parts[0], .scales = parts[1], .biases = parts[2], .bias = b, .bits = 4, .group_size = 32 },
    };
    for (cases) |l| {
        var sc = scope_mod.Scope.init(testing.allocator, s);
        defer sc.deinit();
        const old = scope_mod.Linear{ .w = l.w, .scales = l.scales, .biases = l.biases, .bias = l.bias, .bits = l.bits, .group_size = l.group_size };
        try ops.expectBitEqual(try l.forward(&g, x), try old.forward(&sc, x));
    }
}
