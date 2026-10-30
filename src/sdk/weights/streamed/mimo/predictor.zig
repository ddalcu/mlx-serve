//! The shared scheduler's decode lookahead predictor and profiling surface (MiMo's, from its expert math): the next
//! routed layer's scores `sigmoid(x @ router) + correction` from the F32 transposed router, M one-row products past one
//! row. The scores choose only the lookahead's pre-reads, never an expert, a weight or a slot of a forward.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("../ops.zig");
const profile = @import("profile.zig");

const T = mlx.mlx_array;

/// A prompt wave's rows at most on the control (MiMo's projection tile).
pub const projection_rows: u32 = 1024;

pub const Profile = struct {
    pub const enabled = profile.enabled;
    pub const now = profile.now;
    pub const Mode = @import("../profile/prefill_timers.zig").Mode;
    pub const charge = profile.charge;
    pub fn setMode(_: Mode) void {}
    pub fn countWideCall() void {}
};

pub const Math = struct {
    /// The decode lane takes at most this many rows a call; the wide lane from the next.
    pub const decode_forward_rows: u32 = 4;

    pub fn predictorScores(g: anytype, x: @TypeOf(g.*).T, router: @TypeOf(g.*).T, correction: @TypeOf(g.*).T) !@TypeOf(g.*).T {
        if (comptime @TypeOf(g.*) == ops.MlxOps) {
            const m = g.shapeOf(x).dim(0);
            if (m > 1) return g.add(try g.sigmoid(try predictorRows(g, try g.astype(x, .float32), router, m)), correction);
        }
        return g.add(try g.sigmoid(try g.matmul(try g.astype(x, .float32), router)), correction);
    }

    /// `xf [M, K] @ router [K, N]` as M one-row products in one `gather_mm` (MLX's `gather_mv`).
    fn predictorRows(g: *ops.MlxOps, xf: T, router: T, m: c_int) !T {
        const k = g.shapeOf(xf).dim(1);
        const n = g.shapeOf(router).dim(1);
        const rows = try g.reshape(xf, &.{ m, 1, k });
        const one = try g.reshape(router, &.{ 1, k, n });
        const zero: [1]u32 = .{0};
        const rhs = try g.hostArray(std.mem.sliceAsBytes(&zero), &.{1}, .uint32);
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_gather_mm(&out, rows, one, .{}, rhs, false, g.s)) catch |err| {
            _ = mlx.mlx_array_free(out);
            return err;
        };
        return g.reshape(try g.adopt(out), &.{ m, n });
    }
};

pub const PredictRoute = enum { f32, bf16 };

pub const Predictor = struct {
    scores: *const fn (*const Predictor, *ops.MlxOps, T, T, T) anyerror!T = scoresF32,

    pub fn install(_: PredictRoute, _: anytype) !Predictor {
        return .{};
    }

    pub fn deinit(_: *Predictor) void {}

    fn scoresF32(_: *const Predictor, g: *ops.MlxOps, x: T, router: T, correction: T) anyerror!T {
        return Math.predictorScores(g, x, router, correction);
    }
};
