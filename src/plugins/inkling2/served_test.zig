//! Inkling 2 served through the engine's plugin driver on its pack (`INKLING2_PACK`): a prompt restored from the
//! prefix cache's checkpoints, as the scheduler restores one, decodes from the logits its first run decoded from.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const HotPrefixCache = @import("../../engines/mlx/cache/prefix_cache.zig").HotPrefixCache;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const TokenLogprob = @import("../../sdk/request/logprob.zig").TokenLogprob;

const testing = std.testing;
const TOP = 20;
const Logprobs = [TOP]TokenLogprob;

/// The first token's top logprobs of `prompt[from..]` after `from` rows were restored, as the scheduler's generator
/// takes them (stride-128 checkpoints, greedy); with `hc`, the prompt's rows and checkpoints are committed to it.
fn firstLogprobs(xfm: *Transformer, prompt: []const u32, from: usize, hc: ?*HotPrefixCache) !Logprobs {
    const a = testing.allocator;
    var tok: Tokenizer = undefined;
    var gen = try Generator.initWithOptions(testing.io, a, xfm, &tok, prompt[from..], 1, .{ .temperature = 0.0 }, &.{}, .{ .ssm_checkpoint_stride = 128, .ssm_checkpoint_pos_offset = from, .logprobs_n = TOP });
    defer gen.deinit(a);
    var out: Logprobs = undefined;
    @memcpy(&out, gen.pending_logprob.?.top_logprobs[0..TOP]);
    if (hc) |c| {
        const cps = gen.takeSsmCheckpoints();
        _ = try c.commitWithSsm(&xfm.cache, prompt, false, if (cps.len > 0) cps else null, null, null);
    }
    return out;
}

/// `prompt` cold, from an empty cache; committed to `hc` when given.
fn cold(xfm: *Transformer, prompt: []const u32, hc: ?*HotPrefixCache) !Logprobs {
    try xfm.resetCache();
    return firstLogprobs(xfm, prompt, 0, hc);
}

/// `prompt` restored from `hc` (the rows its lookup matched), the rest forwarded, then committed back to `hc`.
fn warm(xfm: *Transformer, prompt: []const u32, hc: *HotPrefixCache, matched: *usize) !Logprobs {
    try xfm.resetCache();
    const hit = try hc.lookupAndRestore(&xfm.cache, &xfm.moe_seq_offset, xfm.ssm_entries, xfm.s, prompt, false, &.{}, null, null);
    try testing.expect(hit.matched > 0 and hit.matched < prompt.len);
    matched.* = hit.matched;
    return firstLogprobs(xfm, prompt, hit.matched, hc);
}

fn expectSame(what: []const u8, at: usize, want: *const Logprobs, got: *const Logprobs) !void {
    if (!std.mem.eql(u8, std.mem.asBytes(want), std.mem.asBytes(got))) std.debug.print("[inkling2 served] {s} restored at {d}: first logprobs differ\n", .{ what, at });
    try testing.expectEqualSlices(u8, std.mem.asBytes(want), std.mem.asBytes(got));
}

// Bar: an exact repeat takes the first token's top-20 logprobs its first run took, bit for bit: a cold prompt's at two
// lengths, and a prompt's that first restored from another's checkpoint inside their shared prefix.
test "inkling2 pack: a prompt restored from the prefix cache decodes from the logits its first run decoded from, bit for bit (INKLING2_PACK)" {
    const dir = env.get(.INKLING2_PACK) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var cfg = try model.parseConfig(testing.io, a, std.mem.span(dir));
    defer cfg.deinit(a);
    var xfm = try driver.load(testing.io, a, &cfg, std.mem.span(dir), mlx.gpuStream());
    defer xfm.deinit();
    var ids: [1430]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 50000);
    var at: usize = 0;
    for ([_]usize{ 300, ids.len }) |n| {
        var hc = HotPrefixCache.init(a, 4);
        defer hc.deinit();
        const c = try cold(&xfm, ids[0..n], &hc);
        try expectSame("an exact repeat", n, &c, &try warm(&xfm, ids[0..n], &hc, &at));
    }

    // A 231-token prompt leaves its backoff checkpoint at 200; the second prompt shares its first 220 tokens, restores
    // there and forwards the rest, and its repeat restores at its own backoff: the repeat decodes from its logits.
    var other: [1430]u32 = undefined;
    @memcpy(other[0..220], ids[0..220]);
    for (other[220..], 220..) |*t, i| t.* = @intCast((i * 104729 + 7) % 50000);
    var hc = HotPrefixCache.init(a, 4);
    defer hc.deinit();
    _ = try cold(&xfm, ids[0..231], &hc);
    const shared = try warm(&xfm, &other, &hc, &at);
    try testing.expect(at <= 220);
    try expectSame("a shared-prefix prompt's repeat", at, &shared, &try warm(&xfm, &other, &hc, &at));
}
