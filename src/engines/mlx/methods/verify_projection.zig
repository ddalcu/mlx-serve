//! The legacy `Transformer`'s methods over `sdk/blocks/linear/verify_projection.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const Transformer = @import("../transformer.zig").Transformer;
const concatAxis0 = @import("../../../mlx/array.zig").concatAxis0;
const msvQmvRows = @import("../../../sdk/blocks/linear/qmv_rows.zig").msvQmvRows;
const SingleVerifyFeature = @import("../../../sdk/blocks/linear/verify_projection.zig").SingleVerifyFeature;

/// Only qwen4_exp's trunk module runs the single-stream verify arms (its own copy of this gate).
pub fn verifyFeatureEnabled(self: *const Transformer, comptime feature: SingleVerifyFeature, grouped: bool, batch: c_int, width: c_int) bool {
    _ = self;
    _ = feature;
    _ = grouped;
    _ = batch;
    _ = width;
    return false;
}

pub fn mtpProjectionRows(self: *Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lm_head: bool) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (sh.len < 2 or sh.len > 8 or sh[0] < 1 or sh[0] > MAX_BATCH_ROWS) return error.MtpProjectionShape;
    const n: usize = @intCast(sh[0]);
    var virtual_rows: c_int = 1;
    for (sh[1 .. sh.len - 1]) |d| virtual_rows *= d;
    if (virtual_rows == 1) {
        const qp = self.quantParamsHinted(w, sc, lastDim(x));
        if (qp.mode == .affine) {
            if (try msvQmvRows(self.s, x, w, sc, bi, qp.bits, qp.group_size)) |y| return y;
        }
    }
    var parts: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts[0..n]) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    for (0..n) |row| {
        var starts: [8]c_int = @splat(0);
        var stops: [8]c_int = @splat(0);
        const strides: [8]c_int = @splat(1);
        @memcpy(stops[0..sh.len], sh);
        starts[0] = @intCast(row);
        stops[0] = @intCast(row + 1);
        var xrow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrow);
        try mlx.check(mlx.mlx_slice(&xrow, x, &starts, @intCast(sh.len), &stops, @intCast(sh.len), &strides, @intCast(sh.len), self.s));
        parts[row] = if (lm_head) try self.lmHeadProject(xrow) else try self.qmatmul(xrow, w, sc, bi);
    }
    return concatAxis0(self.s, parts[0..n]);
}
