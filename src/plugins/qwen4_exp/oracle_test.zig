//! A qwen4_exp proof's expected values: the tape the legacy forward recorded (`fixtures/<name>.tape`,
//! `engines/mlx/plugin/tape.zig`), and a reference forward's value where the proof has one.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");

const Array = mlx.mlx_array;

pub const Oracle = struct {
    tape: tape_mod.Tape,

    pub fn open(comptime name: []const u8) !Oracle {
        return .{ .tape = try tape_mod.Tape.player(@embedFile("fixtures/" ++ name ++ ".tape")) };
    }

    /// `got` against `want` (a reference forward's, when the proof has one) and the tape; `tokens`: a greedy run,
    /// compared only on the recording's device.
    pub fn check(o: *Oracle, got: Array, want: ?Array, tokens: bool) !void {
        if (want) |w| try ops.expectBitEqual(got, w);
        try o.tape.expectNext(got, tokens);
    }

    /// `check` on host ids (a token run, accepted counts).
    pub fn checkIds(o: *Oracle, got: []const u32, want: ?[]const u32) !void {
        const g = idsArray(got);
        defer _ = mlx.mlx_array_free(g);
        const w: ?Array = if (want) |x| idsArray(x) else null;
        defer if (w) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try o.check(g, w, true);
    }

    /// The playback read every value.
    pub fn close(o: *Oracle) !void {
        try o.tape.expectDone();
    }
};

fn idsArray(ids: []const u32) Array {
    if (ids.len == 0) return mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{0}, 1, .uint32);
    return mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
}
