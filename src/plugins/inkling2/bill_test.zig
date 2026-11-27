//! inkling2's KV bill against the KV its forward holds. The plugin bills by the engine's arithmetic over its shell and
//! layout (`plugin.bill`); this file, outside the plugin's closure, reads that arithmetic (`plugin/config.zig adapted`)
//! and pins it to the caches a prompt fills, under every `loopcd` mode.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const plugin = @import("plugin.zig");
const engine_plugin = @import("../../engines/mlx/plugin/config.zig");

const testing = std.testing;
const prompt = [_]u32{ 1, 17, 42, 55, 3, 9, 7, 30, 44, 5, 22, 31, 58, 2, 13, 49, 27, 8, 36, 19, 51, 4, 40, 11 };

/// `test_cache.ops` noting which entries the forward appends to as a sliding ring, the rest growing whole.
const Seen = struct {
    var ring: [plugin.config.max_visits]bool = undefined;

    fn appendRing(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, window: u32, rows: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        ring[layer] = true;
        return test_cache.ops.append_ring(h, layer, k, v, keep, window, rows, s, out);
    }

    const ops: abi.CacheOps = blk: {
        var o = test_cache.ops;
        o.append_ring = appendRing;
        break :blk o;
    };
};

/// K plus V elements one row adds to an entry, read off the cache the forward filled.
fn rowElems(l: anytype) u64 {
    var sum: u64 = 0;
    for ([_]mlx.mlx_array{ l.k, l.v }) |x| {
        const sh = mlx.getShape(x);
        sum += @as(u64, @intCast(sh[1])) * @as(u64, @intCast(sh[3]));
    }
    return sum;
}

// Bar: after a prompt, every cache entry the engine sizes holds the prompt's rows and its conv state, and the bf16 KV
// the engine bills (whole entries per token, ring entries per ring row) is what the entries the forward grew whole,
// and those it appended as rings, hold.
test "inkling2 bill: the KV billed per token is the KV the forward holds, in every loopcd mode" {
    const a = testing.allocator;
    for ([_][]const u8{
        "",
        "{\"loopcd\":\"hidden\"}",
        "{\"loopcd\":\"logits\"}",
        "{\"loopcd\":{\"mode\":\"adaptive\",\"ref_pass\":2}}",
    }) |settings| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena.allocator()), .{ .settings = settings });
        defer pack.close();
        const n = pack.parsed.shell.num_hidden_layers;
        var cache = try test_cache.TestCache.init(a, n);
        defer cache.deinit();
        var h = [_]abi.Handle{cache.handle()};
        const slot = [_]u32{0};
        const pos = [_]u32{0};
        var out: abi.ForwardOut = .{};
        @memset(&Seen.ring, false);
        try testing.expect(pack.table.forward(pack.model, &.{ .mode = .prefill, .logits = .last_row, .rows = prompt.len, .tokens = .of(&prompt), .slots = .of(&slot), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &Seen.ops }, &out).isOk());
        if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
        a.free(try conformance.consumeF32(a, out.logits));

        var binding: engine_plugin.Binding = .{ .table = &plugin.table, .parsed = pack.parsed };
        binding.readBillTerms();
        const cfg = engine_plugin.adapted(&binding);
        var full: u64 = 0;
        var sliding: u64 = 0;
        for (cache.layers, Seen.ring[0..n]) |l, ring| {
            try testing.expectEqual(@as(u64, prompt.len), l.rows);
            try testing.expect(l.conv.ctx != null);
            if (ring) sliding += rowElems(l) else full += rowElems(l);
        }
        try testing.expectEqual(full * 2, cfg.kvBytesPerToken());
        try testing.expectEqual(sliding * 2, cfg.slidingRowBytes());
    }
}

// Bar: on the release's geometry the engine bills the KV of every entry the plugin's layout declares, in every mode,
// with the reference coda's entries.
test "inkling2 bill: the release config's KV bill is its layout's in every loopcd mode" {
    const sdk = @import("../../sdk/root.zig");
    const config = plugin.config;
    const json = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/mlx-community_Inkling-2-Mini-4bit.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    for ([_][]const u8{ "", "{\"loopcd\":\"hidden\"}", "{\"loopcd\":\"logits\"}", "{\"loopcd\":{\"mode\":\"adaptive\",\"ref_pass\":2}}" }) |settings| {
        var binding: engine_plugin.Binding = .{ .table = &plugin.table };
        try testing.expect(config.parse(&dir, .of(json), .of(settings), null, true, &binding.parsed).isOk());
        defer config.freeParsed(&binding.parsed);
        binding.readBillTerms();
        const cfg = engine_plugin.adapted(&binding);
        var lay: [config.max_visits]abi.LayerCache = undefined;
        const n = config.layout(&binding.parsed, null, &lay, config.max_visits);
        try testing.expectEqual(binding.parsed.shell.num_hidden_layers, n);
        var full: u64 = 0;
        var sliding: u64 = 0;
        for (lay[0..n]) |l| {
            const row: u64 = @as(u64, l.kv_heads) * (l.k_dim + l.v_dim) * 2;
            if (l.kind == .full) full += row else sliding += row;
        }
        try testing.expectEqual(full, cfg.kvBytesPerToken());
        try testing.expectEqual(sliding, cfg.slidingRowBytes());
    }
}
