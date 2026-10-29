//! The streamed-experts contract (`streamed.zig` `Streamed.routedSource`) a trunk's MoE reads, free of the streamer
//! itself so any MoE block can carry it.

const mlx = @import("../../mlx/ffi.zig");
const err_name = @import("../util/err_name.zig");

/// One projection's slot-bank arrays as the trunk's gathers read them (MXFP4: words and E8M0 scales; affine: words,
/// scales and biases).
pub const Proj = struct { w: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array = .{ .ctx = null } };
pub const Banks = struct { gate: Proj, up: Proj, down: Proj };
/// A call's slots: per routed pair (the router's flat order) its bank (prompt rows, grown rows, transient rows) and its
/// row there, and each bank's arrays. Valid until the next fetch.
pub const Fetched = struct { kind: []const u8, row: []const u32, banks: [3]?Banks };
/// Routed pairs one call takes at most.
pub const max_call_ids: usize = 1 << 14;
/// Distinct experts one fetch takes at most (the expert stream's route width).
pub const max_call_experts: usize = 48;

/// How a trunk's MoE reads a streamed bank, by whose math runs it. `trunk`: the source fetches, `fetch` making model
/// layer `layer`'s routed experts `ids` resident and naming each pair's slot (a decode call's `scores`, the next
/// layer's predictor on its rows, `[rows * n_experts]`, start that layer's likely reads; empty: none), and the trunk
/// runs its own gathers on them, so its weighted rows are its own op order. `source`: a format the trunk has no math for (EXL3), the source's
/// routed sum `[n, hidden]` f32 for the normed input, the routes and their weights. `settle` after the eval that
/// consumed a forward's calls.
pub const Routed = struct {
    ctx: *anyopaque,
    math: union(enum) {
        trunk: *const fn (*anyopaque, u32, []const u16, []const f32) Ret(Fetched),
        source: *const fn (*anyopaque, u32, mlx.mlx_array, mlx.mlx_array, mlx.mlx_array) Ret(mlx.mlx_array),
    },
    settle_fn: *const fn (*anyopaque) Ret(void),
    /// A source whose slot rows follow the pass (`true`: a prompt pass, `false`: decode), told before each forward.
    phase_fn: ?*const fn (*anyopaque, bool) Ret(void) = null,
    /// A prompt pass's predicted routes of model layer `layer` (every row's ids), read while its attention runs.
    read_ahead_fn: ?*const fn (*anyopaque, u32, []const u16) Ret(void) = null,
    /// Routed ids a decode-phase call takes: a wider call is a prompt pass, which a source refuses in decode.
    decode_ids: usize = max_call_ids,

    pub fn settle(r: Routed) !void {
        return r.settle_fn(r.ctx).get();
    }

    pub fn phase(r: Routed, prompt: bool) !void {
        if (r.phase_fn) |f| try f(r.ctx, prompt).get();
    }
};

/// A `Routed` call's result. The source is built in the plugin's compile unit and may be called from another (a
/// DSpark round runs in the modules unit), where an `anyerror` value names a different error: it crosses by name.
pub fn Ret(comptime T: type) type {
    return struct {
        val: T = undefined,
        err: []const u8 = "",

        pub fn of(r: anyerror!T) @This() {
            return if (r) |v| .{ .val = v } else |e| .{ .err = @errorName(e) };
        }
        pub fn get(self: @This()) err_name.Known!T {
            if (self.err.len != 0) return err_name.toError(self.err);
            return self.val;
        }
    };
}

test "routed: an error crosses by name" {
    const std = @import("std");
    const r = Ret(u32).of(error.ExpertReadShort);
    try std.testing.expectError(error.EngineFailed, r.get());
    try std.testing.expectEqualStrings("ExpertReadShort", err_name.last());
    try std.testing.expectEqual(@as(u32, 7), try Ret(u32).of(7).get());
}
