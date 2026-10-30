//! mlx-stream's serial sampler: the plugin's token for every row it hands the host. Every arch binds one per module.
//! The host arms a clean request through `draft_lane.arm` before its prompt pass (`arm`). The prompt pass takes the
//! binding (`begin`): a request the host did not arm (logprobs, grammar, penalties, `--no-mtp`) is unbound and keeps
//! the host's sampler.
//!
//! For a bound sampled request every logits row the arch returns becomes the one-hot of the token `acceptance` draws
//! at that position (`chosen`): the prompt's last row (the first token) and each serial step's. That token is
//! `acceptance.sample`: the row's tempered, filtered distribution, drawn at `drawAt(seed, position, token_draw)`. A
//! lane's verify emits the same token at the same position (`sampledRows`, sample-then-verify), so a seeded request
//! yields the same ids serial or drafted, on every arch. The host's own sampler then returns the plugin's token
//! whatever its stream. A greedy request keeps its logits (the host's argmax is the target's).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const acc = @import("acceptance.zig");

/// The sampler over an arch's op backend `G` (its `init(gpa, stream)`, `reset`, `keep` and the acceptance graphs'
/// ops), its own scope reset after each row.
pub fn SamplerOf(comptime G: type) type {
    return struct {
        const Self = @This();
        g: G,
        /// The armed request's sampling, until its prompt pass (`begin`).
        pending: ?acc.Params = null,
        /// The request's: null = unbound or greedy (the logits go to the host as they are).
        active: ?acc.Params = null,

        pub fn init(gpa: std.mem.Allocator, s: mlx.mlx_stream) !Self {
            return .{ .g = try G.init(gpa, s) };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        /// `draft_lane.arm`: a clean sampled request's sampling, for its prompt pass (greedy: the host's argmax).
        pub fn arm(self: *Self, req: sdk.ArmRequest) void {
            self.pending = if (req.clean and !req.greedy) bindingOf(req.sampling) else null;
        }

        /// The prompt pass's start: the request's binding (none when the host did not arm it). A bound request logs
        /// once, with the decode lane it runs on (`lane`: the arch's lane name, "serial" without one).
        pub fn begin(self: *Self, lane: []const u8) void {
            self.active = self.pending;
            self.pending = null;
            if (self.active) |sm| sdk.log.info("mlx-stream sampler: request bound (the plugin draws every token) lane=\"{s}\" seed={d} temperature={d:.3} top_p={d:.3} top_k={d}\n", .{ lane, sm.seed, sm.temperature, sm.top_p, sm.top_k });
        }

        /// `logits` (owned by the caller; its last axis the vocabulary) predicts the token at absolute position
        /// `pos`. Bound and sampled: a new owned f32 [1, vocab], 0 at the drawn token and -inf elsewhere (`logits`
        /// freed). Otherwise `logits` itself.
        pub fn chosen(self: *Self, logits: mlx.mlx_array, pos: u64) !mlx.mlx_array {
            const sm = self.active orelse return logits;
            const g = &self.g;
            defer g.reset();
            const sh = g.shapeOf(logits);
            const vocab = sh.dim(-1);
            const row = try g.reshape(logits, &.{ 1, vocab });
            const sr = try acc.sampledRows(g, row, sm, pos - 1, &.{});
            const rank = try g.arange(0, @floatFromInt(vocab), 1, .uint32);
            const hot = try g.where(try g.equal(rank, sr.tok), try g.scalar(0, .float32), try g.scalar(-std.math.inf(f64), .float32));
            const out = g.keep(try g.reshape(hot, &.{ 1, vocab }));
            _ = mlx.mlx_array_free(logits);
            return out;
        }
    };
}

/// A request's binding: its sampling when sampled, null when greedy (the host's argmax).
pub fn bindingOf(s: sdk.SamplingParams) ?acc.Params {
    return acc.Params.active(acc.Params.of(s));
}

const testing = std.testing;

/// Seeded rows: each a leader over a spread of logits.
fn fixture(seed: u64, comptime w: usize, comptime v: usize, lg: *[w * v]f32) void {
    var rng: std.Random.DefaultPrng = .init(seed);
    for (lg) |*x| x.* = rng.random().floatNorm(f32) * 1.2;
    for (0..w) |r| lg[r * v + rng.random().uintLessThan(u32, @as(u32, v))] += 2.0;
}

fn argmaxRow(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = @intCast(i);
    };
    return best;
}

test "sdk_ext sampler: one stream: the same (seed, position, index) is the same draw everywhere; positions, seeds and indices differ" {
    // One implementation for every arch (each binds `SamplerOf` and its lanes call `acceptance`): pinned values.
    try testing.expectEqual(acc.drawAt(1, 1000, acc.token_draw), acc.drawV(1, 1000));
    try testing.expectEqual(acc.drawAt(1, 1000, acc.test_draw), acc.drawU(1, 1000));
    try testing.expectEqual(acc.drawAt(1, 1000, 0), acc.drawAt(1, 1000, 0));
    var seen: [64]f32 = undefined;
    for (&seen, 0..) |*u, p| u.* = acc.drawAt(1, 4096 + p, acc.token_draw);
    for (seen, 0..) |u, i| for (seen[i + 1 ..]) |w| try testing.expect(u != w);
    try testing.expect(acc.drawAt(1, 7, 0) != acc.drawAt(2, 7, 0) and acc.drawAt(1, 7, 0) != acc.drawAt(1, 7, 1));
    // The derivation of record (splitmix64, the lane's since 2026-09; its top 24 bits): a change re-keys every seeded
    // request on every path at once.
    for ([_][4]u64{ .{ 1, 0, 0, 120_324 }, .{ 1, 1, 0, 4_735_337 }, .{ 1, 16_385, 0, 7_294_628 }, .{ 1, 16_385, 1, 9_557_175 } }) |c|
        try testing.expectEqual((@as(f32, @floatFromInt(c[3])) + 0.5) / 16777216.0, acc.drawAt(c[0], c[1], @intCast(c[2])));
}

test "sdk_ext sampler: a seeded request yields the same ids serial, through a lane's verify (drafts accepted or rejected) and through DeepSeek-V4.1's per-row step, sampled and greedy" {
    // The host reference of each path (the device graphs are `sampledRows` / `decisionGraph`, checked against these by
    // the arches' MLX parity tests): serial = `sample` per position (the first token included); a lane = `accept` over
    // [t1, drafts] (glm5_next's and MiMo's `decisionGraph`, GLM-5.3's MTP); DeepSeek-V4.1 = `specStep` row by row (its
    // `decideSampled` oracle).
    const W = 9;
    const V = 96;
    var lg: [W * V]f32 = undefined;
    fixture(0x5eed, W, V, &lg);
    const base: u64 = 16_384;
    for ([_]acc.Params{ .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 }, .{ .temperature = 0.0, .seed = 1 } }) |sm| {
        // Serial: row r's token at base + r + 1.
        var serial: [W]u32 = undefined;
        var d: [V]f64 = undefined;
        for (&serial, 0..) |*t, r| {
            const row = lg[r * V ..][0..V];
            t.* = if (acc.Params.active(sm)) |s| blk: {
                acc.distribution(row, s, &d);
                break :blk acc.sample(&d, s.seed, base + r + 1);
            } else argmaxRow(row);
        }
        // A lane: rounds of up to 3 drafts from the serial position; drafts right, then one wrong, then right again.
        var lane: [W]u32 = undefined;
        var n: usize = 0;
        var k_round: usize = 0;
        while (n < W - 1) : (k_round += 1) {
            const k: usize = @min(3, W - 1 - n);
            var drafts: [3]u32 = undefined;
            for (drafts[0..k], 0..) |*x, i| x.* = if (k_round % 2 == 1 and i == 1) (serial[n + i] + 1) % V else serial[n + i];
            const r = acc.accept(.exact, sm, lg[n * V ..][0 .. (k + 1) * V], drafts[0..k], null, base + n);
            for (0..r.n_accepted) |i| lane[n + i] = drafts[i];
            lane[n + r.n_accepted] = r.next_token;
            n += r.n_accepted + 1;
            if (k_round % 2 == 1) try testing.expectEqual(@as(u32, 1), r.n_accepted);
        }
        try testing.expectEqualSlices(u32, serial[0..n], lane[0..n]);
        // DeepSeek-V4.1's step: each row's token, accepted when it is the draft.
        for (0..W) |r| {
            const row = lg[r * V ..][0..V];
            if (acc.Params.active(sm)) |s| {
                acc.distribution(row, s, &d);
                const st = acc.specStep(&d, serial[r], null, acc.drawU(s.seed, base + r + 1), acc.drawV(s.seed, base + r + 1));
                try testing.expect(st.accept and st.tok == serial[r]);
                const miss = acc.specStep(&d, (serial[r] + 1) % V, null, acc.drawU(s.seed, base + r + 1), acc.drawV(s.seed, base + r + 1));
                try testing.expect(!miss.accept and miss.tok == serial[r]);
            } else try testing.expectEqual(serial[r], argmaxRow(row));
        }
    }
}

test "sdk_ext sampler: arm binds a clean sampled request for its prompt pass; greedy, unclean and unarmed requests stay the host's" {
    const S = SamplerOf(NoOps);
    var s: S = .{ .g = .{} };
    s.arm(.{ .greedy = false, .clean = true, .sampling = .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 } });
    s.begin("serial");
    try testing.expectEqual(@as(u64, 1), s.active.?.seed);
    // The next request is not armed: unbound.
    s.begin("serial");
    try testing.expect(s.active == null);
    s.arm(.{ .greedy = true, .clean = true, .sampling = .{ .temperature = 0.0, .seed = 1 } });
    s.begin("serial");
    try testing.expect(s.active == null);
    s.arm(.{ .greedy = false, .clean = false, .sampling = .{ .temperature = 1.0, .seed = 1 } });
    s.begin("serial");
    try testing.expect(s.active == null);
}

/// The binding's bookkeeping needs no ops (the CPU test above): an empty backend.
const NoOps = struct {};
