//! The server's own pack formats behind the KDA-hybrid trunk's hooks (`trunk.Hooks`), for core's two plugins
//! (`glm5_next`, `bailing_hybrid`): Sushi's EXL3 routed experts (`expert_quant`). The kit's `kda_hybrid` index never
//! names this file: it imports the EXL3 format and so the `sushi_exl3` module, which the kit does not ship.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const quant_abi = @import("../../../abi/quant.zig");
const exl3_format = @import("../../quant/exl3/format.zig");
const trunk = @import("trunk.zig");

const Exl3 = trunk.Hooks.Exl3;
const Array = mlx.mlx_array;

comptime {
    // The trunk carries the format's spec inline.
    if (@sizeOf(exl3_format.Spec) > Exl3.spec_bytes or @alignOf(exl3_format.Spec) > 16) @compileError("trunk.Hooks.Exl3.Spec cannot hold exl3.Spec");
}

fn specOf(sp: *const Exl3.Spec) *const exl3_format.Spec {
    return @ptrCast(@alignCast(&sp.bytes));
}

fn parse(gpa: std.mem.Allocator, expert_quant_json: []const u8, top_k: u32, out: *Exl3.Spec) anyerror!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const text = try std.fmt.allocPrint(arena.allocator(), "{{\"expert_quant\":{s}}}", .{expert_quant_json});
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
    const spec = try exl3_format.parse(root.object, top_k);
    out.* = .{};
    @as(*exl3_format.Spec, @ptrCast(@alignCast(&out.bytes))).* = spec;
}

fn admit(sp: *const Exl3.Spec, trellis: Array, suh: Array, svh: Array, experts: u32, in_dim: u32, out_dim: u32) anyerror!void {
    try exl3_format.admit(.{ .trellis = trellis, .suh = suh, .svh = svh }, experts, in_dim, out_dim, specOf(sp).*);
}

/// A Sushi bank as the EXL3 format table loads one: trellis, suh, svh and the pack's decode.
fn wrap(sp: *const Exl3.Spec, trellis: Array, suh: Array, svh: Array) quant_abi.Weight {
    const d = exl3_format.decode(specOf(sp).*);
    return .{
        .w = .{ .ctx = trellis.ctx },
        .scales = .{ .ctx = suh.ctx },
        .biases = .{ .ctx = svh.ctx },
        .layout = @backingInt(d.codebook),
        .flags = @intCast(d.window.bits()),
        .block = @intCast(mlx.getShape(trellis)[3]),
        .state = if (d.table) |i| @ptrFromInt(@as(usize, i) + 1) else null,
    };
}

/// The trunk's EXL3 hook for core's plugins.
pub const exl3: Exl3 = .{ .parse = &parse, .admit = &admit, .wrap = &wrap, .table = &exl3_format.table };

/// The spec as the format reads it, for this file's tests.
pub fn formatSpec(sp: *const Exl3.Spec) exl3_format.Spec {
    return specOf(sp).*;
}
