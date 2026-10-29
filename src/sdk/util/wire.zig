//! A two-Mac group's wire protocol (fixed int32[8] control messages, then any payload tensor) and the link a plugin's
//! split job talks over, as the engine's cluster hands it (`engines/mlx/cluster/distributed.zig`).

const mlx = @import("../../mlx/ffi.zig");

pub const MAGIC: i32 = 0x4D4C5853; // "MLXS"
pub const Msg = [8]i32;
pub const OP_STOP: i32 = 2;
pub const OP_H3_JOB: i32 = 4;
pub const OP_H3_STEP: i32 = 5;
pub const OP_H3_END: i32 = 6;
/// The host unloaded the pair's model: the worker frees its part and waits for the next plan.
pub const OP_RELEASE: i32 = 8;

/// The other rank of a two-Mac group.
pub const Link = struct {
    ctx: *const anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        send_op: *const fn (ctx: *const anyopaque, m: Msg) anyerror!void,
        recv_op: *const fn (ctx: *const anyopaque) anyerror!Msg,
        send_now: *const fn (ctx: *const anyopaque, x: mlx.mlx_array, dst: u32) anyerror!void,
        recv_now: *const fn (ctx: *const anyopaque, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) anyerror!mlx.mlx_array,
        all_gather: *const fn (ctx: *const anyopaque, x: mlx.mlx_array) anyerror!mlx.mlx_array,
    };

    /// A control message to the other rank.
    pub fn sendOp(self: *const Link, m: Msg) !void {
        return self.vt.send_op(self.ctx, m);
    }

    pub fn recvOp(self: *const Link) !Msg {
        return self.vt.recv_op(self.ctx);
    }

    /// `x` sent and complete before this returns.
    pub fn sendNow(self: *const Link, x: mlx.mlx_array, dst: u32) !void {
        return self.vt.send_now(self.ctx, x, dst);
    }

    pub fn recvNow(self: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
        return self.vt.recv_now(self.ctx, shape, dtype, src);
    }

    /// Every rank's `x` stacked on axis 0 in rank order; the shapes must match.
    pub fn allGather(self: *const Link, x: mlx.mlx_array) !mlx.mlx_array {
        return self.vt.all_gather(self.ctx, x);
    }
};
