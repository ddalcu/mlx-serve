//! The engine's `CacheOps` over its `KVCache` and SSM entries (`sdk/kv/cache_ops.zig`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const abi = @import("../../../abi/plugin.zig");
const KVCache = @import("cache.zig").KVCache;
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const adapter = @import("../../../sdk/kv/cache_ops.zig").For(SSMCacheEntry);

pub const Slot = adapter.Slot;
pub const ops = adapter.ops;

fn mx(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

fn ab(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

// ── tests ──

const testing = std.testing;
const conformance = @import("../../../sdk/testing/cache_ops.zig");

fn newEntries(n: usize) ![]SSMCacheEntry {
    const es = try testing.allocator.alloc(SSMCacheEntry, n);
    for (es) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return es;
}

fn freeEntries(es: []SSMCacheEntry) void {
    for (es) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
    }
    testing.allocator.free(es);
}

test "bases cache ops: the KVCache adapter passes the CacheOps conformance checks" {
    var c1 = try KVCache.init(testing.allocator, conformance.LAYERS);
    defer c1.deinit();
    var c2 = try KVCache.init(testing.allocator, conformance.LAYERS);
    defer c2.deinit();
    const e1 = try newEntries(conformance.LAYERS);
    defer freeEntries(e1);
    const e2 = try newEntries(conformance.LAYERS);
    defer freeEntries(e2);
    var s1: Slot = .{ .kv = &c1, .ssm = e1 };
    var s2: Slot = .{ .kv = &c2, .ssm = e2 };
    try conformance.expectConforms(&ops, s1.handle(), s2.handle());
}

// Bar: a quantized cache hands the packed triples to the view and owns the dequantized arrays it returns.
test "bases cache ops: a quantized KVCache's view carries its triples" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var c = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer c.deinit();
    var sl: Slot = .{ .kv = &c };
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, 2, 3, 64 }, 4, .bfloat16, s));
    var out: abi.KvView = .{};
    try testing.expect(ops.append(sl.handle(), 0, ab(k), ab(k), 0, .{ .ctx = s.ctx }, &out).isOk());
    defer ops.release(&out);
    try testing.expect(out.owned and out.bits == 8 and out.group_size == 64);
    try testing.expect(out.k_q.ctx != null and out.v_biases.ctx != null and out.capacity >= 3);
    try testing.expectEqualSlices(c_int, &.{ 1, 2, 3, 64 }, mlx.getShape(mx(out.k)));
    try testing.expectEqual(@as(u64, 3), ops.len(sl.handle(), 0));
}
