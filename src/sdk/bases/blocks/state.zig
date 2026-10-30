//! A recurrent layer's state across the forward's slots: one slot's state as it is, N slots' stacked on the batch
//! axis through `CacheOps.state_rows` (an engine may keep them merged across ticks). Blocks read and write recurrent
//! state only through these.

const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const block = @import("../block.zig");

const Array = ops.Array;
const Ctx = block.Ctx;

/// The current layer's `slot` over every slot of the forward (`CacheOps.state_rows` past one slot), owned by `c.g`;
/// null when no slot holds it yet. A batch where only some slots hold it is refused (their widths would disagree).
pub fn get(c: *const Ctx, slot: abi.StateSlot) !?Array {
    const o = try c.cacheOps();
    if (c.caches.len == 1) {
        const a = o.state(c.caches[0], c.layer, slot);
        return if (a.ctx == null) null else try c.g.keep(.{ .ctx = a.ctx });
    }
    const rows = o.state_rows(c.caches.ptr, @intCast(c.caches.len), c.layer, slot, .{ .ctx = c.stream().ctx });
    if (rows.ctx != null) return try c.g.adopt(.{ .ctx = rows.ctx });
    for (c.caches) |h| if (o.state(h, c.layer, slot).ctx != null) return error.BatchedStateUnready;
    return null;
}

/// Writes `value` as the current layer's `slot`, each forward slot taking its row.
pub fn put(c: *const Ctx, slot: abi.StateSlot, value: Array) !void {
    const o = try c.cacheOps();
    if (c.caches.len == 1) return ok(o.set_state(c.caches[0], c.layer, slot, .{ .ctx = value.ctx }));
    return ok(o.set_state_rows(c.caches.ptr, @intCast(c.caches.len), c.layer, slot, .{ .ctx = value.ctx }, .{ .ctx = c.stream().ctx }));
}

fn ok(v: abi.Verdict) !void {
    if (!v.isOk()) return error.CacheOpFailed;
}
