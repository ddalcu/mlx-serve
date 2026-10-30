//! A trunk weight as a drafter reads it (`SpecHooks.trunk_weight`): borrowed arrays, the weight's own geometry and
//! layout.

const abi = @import("../../../abi/plugin.zig");
const quant_abi = @import("../../../abi/quant.zig");
const native = @import("../../quant/native.zig");
const Linear = @import("linear.zig").Linear;

/// `l` borrowed into `out`.
pub fn trunkWeight(l: *const Linear, out: *quant_abi.Weight) void {
    const layout: native.Layout = if (l.s.ctx == null) .dense else switch (l.mode) {
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
        .mxfp8 => .mxfp8,
        else => .affine,
    };
    out.* = .{ .w = .{ .ctx = l.w.ctx }, .scales = .{ .ctx = l.s.ctx }, .biases = .{ .ctx = l.b.ctx }, .bits = @intCast(l.bits), .group_size = l.group_size, .layout = @intFromEnum(layout) };
}

/// `SpecHooks.trunk_weight` over a loaded model whose base holds a `TableEmbed` and an `LmHead`.
pub fn trunkWeightHook(comptime Loaded: type) *const fn (*anyopaque, abi.TrunkWeight, *quant_abi.Weight) callconv(.c) abi.Verdict {
    return &struct {
        fn f(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
            const m: *Loaded = @ptrCast(@alignCast(mp));
            trunkWeight(switch (which) {
                .embed => &m.base.weights.embed.table,
                .head => &m.base.weights.head,
            }, out);
            return .ok;
        }
    }.f;
}
