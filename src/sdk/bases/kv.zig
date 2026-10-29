//! A block's cache calls: `CacheOps` verdicts as errors, and a view in the shape the shared attention kernels read.

const abi = @import("../../abi/plugin.zig");
const block = @import("block.zig");
const DenseKVView = @import("../blocks/attention/kv_view.zig").DenseKVView;

const Array = block.Array;
const Ctx = block.Ctx;

fn ok(v: abi.Verdict) !void {
    if (!v.isOk()) return error.CacheOpFailed;
}

fn ab(a: Array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn mx(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

/// Appends `k`/`v` at the current layer of the solo slot; the view reads the newest `keep` rows (0 = all).
pub fn append(c: *const Ctx, k: Array, v: Array, keep: u32) !abi.KvView {
    const o = try c.cacheOps();
    var out: abi.KvView = .{};
    try ok(o.append(c.cache(), c.layer, ab(k), ab(v), keep, .{ .ctx = c.stream().ctx }, &out));
    return out;
}

/// The rows `layer` of `cache` holds.
pub fn view(c: *const Ctx, cache: abi.Handle, layer: u32) !abi.KvView {
    const o = try c.cacheOps();
    var out: abi.KvView = .{};
    try ok(o.view(cache, layer, .{ .ctx = c.stream().ctx }, &out));
    return out;
}

/// Appends row `i` of the stacked `k`/`v` to slot `i`'s cache at the current layer.
pub fn appendRows(c: *const Ctx, k: Array, v: Array, keep: u32) !void {
    const o = try c.cacheOps();
    try ok(o.append_rows(c.caches.ptr, @intCast(c.caches.len), c.layer, ab(k), ab(v), keep, .{ .ctx = c.stream().ctx }));
}

pub fn release(c: *const Ctx, v: *abi.KvView) void {
    if (c.cache_ops) |o| o.release(v);
}

/// `v` as the shared attention kernels take it (its arrays stay the view's).
pub fn dense(v: abi.KvView) DenseKVView {
    return .{
        .k = mx(v.k),
        .v = mx(v.v),
        .owned = false,
        .k_triple_q = mx(v.k_q),
        .k_triple_scales = mx(v.k_scales),
        .k_triple_biases = mx(v.k_biases),
        .v_triple_q = mx(v.v_q),
        .v_triple_scales = mx(v.v_scales),
        .v_triple_biases = mx(v.v_biases),
        .has_quant_triple = v.bits != 0,
        .bits = v.bits,
        .group_size = v.group_size,
        .kv_cap = @intCast(v.capacity),
    };
}

/// The current layer's recurrent state of the solo slot, borrowed; null before its first write.
pub fn state(c: *const Ctx, slot: abi.StateSlot) !?Array {
    const o = try c.cacheOps();
    const a = o.state(c.cache(), c.layer, slot);
    return if (a.ctx == null) null else mx(a);
}

/// Replaces the current layer's recurrent state (the cache takes its own reference).
pub fn setState(c: *const Ctx, slot: abi.StateSlot, value: Array) !void {
    const o = try c.cacheOps();
    try ok(o.set_state(c.cache(), c.layer, slot, ab(value)));
}

/// `x` owned by the layer's graph, or null.
pub fn own(c: *const Ctx, x: ?Array) !?Array {
    return if (x) |a| try c.g.adopt(a) else null;
}
