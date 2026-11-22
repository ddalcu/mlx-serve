//! The inkling2 table against itself on its tiny looped pack: a prompt forwarded whole, in chunks and token by token
//! serves the same rows, past the sliding window, the relative bias's extent and the log-scaling floor, so every
//! visit's KV and conv windows carry, the one-row sink path agrees with the wide one and the loop's passes keep
//! their own state. On the GPU stream the one-row step runs its kernels (`kernels.zig`), so the op chain of the wide
//! forward is their reference. Not a numeric proof against mlx-lm: that oracle (`models/inkling2`, an open PR) is
//! still owed.

const std = @import("std");
const plugin = @import("plugin.zig");
const draft = @import("draft.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const parity = @import("../../sdk/testing/parity.zig");

const testing = std.testing;
const VOCAB = 60;
const prompt = [_]u32{ 1, 17, 42, 55, 3, 9, 7, 30, 44, 5, 22, 31, 58, 2, 13, 49, 27, 8, 36, 19, 51, 4, 40, 11 };

/// One forward of `tokens` from `pos` through the slot's cache; its logits rows as f32.
fn step(a: std.mem.Allocator, pack: *conformance.Loaded, cache: *test_cache.TestCache, tokens: []const u32, pos: usize, logits: abi.Logits) ![]f32 {
    var h = [_]abi.Handle{cache.handle()};
    const slot = [_]u32{0};
    const positions = [_]u32{@intCast(pos)};
    var out: abi.ForwardOut = .{};
    const mode: abi.Mode = if (tokens.len == 1) .decode else .prefill;
    const v = pack.table.forward(pack.model, &.{ .mode = mode, .logits = logits, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&slot), .positions = .of(&positions), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

/// Bar: the stream is bf16 between visits, so two paths whose f32 sums differ in their last bits (a kernel against its
/// op chain) round some elements apart: cosine above 0.999 and no row whose margin decides it picks another token.
fn expectRows(ours: []const f32, ref: []const f32) !void {
    try testing.expectEqual(ref.len, ours.len);
    const rows = ref.len / VOCAB;
    const cmp = parity.qwen4CompareRows(ours, ref, rows, VOCAB);
    const misses = parity.decidedMisses(ours, ref, rows, VOCAB);
    if (cmp.min_cos <= 0.999 or misses > 0) std.debug.print("[inkling2 proof] {d} rows: min cos {d:.6}, decided misses {d}\n", .{ rows, cmp.min_cos, misses });
    try testing.expect(cmp.min_cos > 0.999 and misses == 0);
}

const Setup = struct {
    arena: std.heap.ArenaAllocator,
    pack: conformance.Loaded,
    whole: []f32,

    /// The pack, on the CPU's stream or the GPU's, and the whole prompt's rows from one forward.
    fn init(a: std.mem.Allocator, gpu: bool) !Setup {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena.allocator()), .{ .gpu = gpu });
        errdefer pack.close();
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        return .{ .arena = arena, .pack = pack, .whole = try step(a, &pack, &cache, &prompt, 0, .all_rows) };
    }

    fn deinit(s: *Setup, a: std.mem.Allocator) void {
        a.free(s.whole);
        s.pack.close();
        s.arena.deinit();
    }
};

/// The streams a proof runs on: the CPU's, and the GPU's where there is a real one.
fn streams() []const bool {
    return if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) &.{false} else &.{ false, true };
}

// On the CPU's stream, where the step's chain and the wide forward's are the same sums: the GPU's f32 matmul is good
// to about 1e-3, and on this random pack one such bit flips an expert pick somewhere along twelve steps.
test "inkling2 proof: a prompt in chunks and then token by token serves the rows of one forward" {
    const a = testing.allocator;
    var s = try Setup.init(a, false);
    defer s.deinit(a);
    var cache = try test_cache.TestCache.init(a, s.pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    var at: usize = 0;
    for ([_]usize{ 5, 7 }) |n| {
        const rows = try step(a, &s.pack, &cache, prompt[at .. at + n], at, .all_rows);
        defer a.free(rows);
        try expectRows(rows, s.whole[at * VOCAB .. (at + n) * VOCAB]);
        at += n;
    }
    while (at < prompt.len) : (at += 1) {
        const row = try step(a, &s.pack, &cache, prompt[at .. at + 1], at, .last_row);
        defer a.free(row);
        try expectRows(row, s.whole[at * VOCAB .. (at + 1) * VOCAB]);
    }
}

test "inkling2 proof: a step reads every visit's conv windows and each loop pass's own KV" {
    for (streams()) |gpu| try stateMutations(gpu);
}

fn stateMutations(gpu: bool) !void {
    const a = testing.allocator;
    var s = try Setup.init(a, gpu);
    defer s.deinit(a);
    const split = 12;
    const want = s.whole[split * VOCAB .. (split + 1) * VOCAB];
    const Mutation = enum { none, no_conv_state, passes_share_kv };
    for ([_]Mutation{ .none, .no_conv_state, .passes_share_kv }) |m| {
        var cache = try test_cache.TestCache.init(a, s.pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        a.free(try step(a, &s.pack, &cache, prompt[0..split], 0, .last_row));
        switch (m) {
            .none => {},
            .no_conv_state => for (cache.layers) |*l| {
                _ = mlx.mlx_array_free(l.conv);
                l.conv = .{};
            },
            // Visits 2..5 are the loop's first pass, 6..9 its second: the second pass given the first's keys.
            .passes_share_kv => for (2..6) |v| {
                try mlx.check(mlx.mlx_array_set(&cache.layers[v + 4].k, cache.layers[v].k));
                try mlx.check(mlx.mlx_array_set(&cache.layers[v + 4].v, cache.layers[v].v));
            },
        }
        const row = try step(a, &s.pack, &cache, prompt[split .. split + 1], split, .last_row);
        defer a.free(row);
        const cos = parity.qwen4CompareRows(row, want, 1, VOCAB).min_cos;
        if (m == .none) try testing.expect(cos > 0.999) else try testing.expect(cos < 0.99);
    }
}

// Bar: an image's rows reach the forward: the same prompt over two images' rows serves different last rows, and
// the same rows the same one.
test "inkling2 proof: media rows splice at the image placeholder" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const config = @import("config.zig");
    // The fixture with an image tower declared (its placeholder inside the tiny vocab) and none shipped.
    var fx = try plugin.conformanceFixture(arena.allocator());
    const files = try arena.allocator().dupe(@TypeOf(fx.files[0]), fx.files);
    files[0].bytes = "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":2,\"image_token_id\":59,\"vision_config\":{\"hidden_size\":32,\"num_hidden_layers\":1,\"num_attention_heads\":2}," ++ config.fixture_text ++ "}";
    fx.files = files;
    var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, fx, .{ .gpu = false });
    defer pack.close();
    try testing.expect(pack.parsed.caps.images and pack.parsed.placeholder_ids.image == 59);
    const tokens = [_]u32{ 1, 17, 59, 59, 3 };
    var rows: [3][]f32 = undefined;
    for (&rows, [_]u64{ 5, 6, 5 }) |*r, seed| {
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        const data = try fixture.randomF32(a, seed, 2 * 64, 1.0);
        defer a.free(data);
        const media = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ 1, 2, 64 }, 3, .float32);
        defer _ = mlx.mlx_array_free(media);
        var h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        const v = pack.table.forward(pack.model, &.{ .mode = .prefill, .logits = .last_row, .rows = tokens.len, .tokens = .of(&tokens), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops, .media = .{ .ctx = media.ctx } }, &out);
        try testing.expect(v.isOk());
        r.* = try conformance.consumeF32(a, out.logits);
    }
    defer for (rows) |r| a.free(r);
    try testing.expectEqualSlices(f32, rows[0], rows[2]);
    try testing.expect(parity.qwen4CompareRows(rows[0], rows[1], 1, VOCAB).min_cos < 0.999);
}

// Bar: a draft round's trunk half. A block of rows verified without appending, then its first rows committed, leaves
// the caches as a forward of those rows alone does: the next step serves the same row (cosine above 0.999), for
// every count kept; and before the commit the caches hold nothing of the block. (Rows against one-row steps is the
// CPU proof above: on the GPU a last bit flips a pick of this fixture.)
test "inkling2 proof: a verified block's kept rows leave the state a forward of those rows leaves" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    const m: *plugin.Loaded = @ptrCast(@alignCast(s.pack.model));
    const run_mod = @import("../../sdk/bases/run.zig");
    const visits = s.pack.parsed.shell.num_hidden_layers;
    const split = 9;
    const width = 6;
    for ([_]usize{ 1, 3, width }) |kept| {
        // The prompt, then the kept rows as one forward, then the step under test.
        var serial = try test_cache.TestCache.init(a, visits);
        defer serial.deinit();
        a.free(try step(a, &s.pack, &serial, prompt[0..split], 0, .last_row));
        a.free(try step(a, &s.pack, &serial, prompt[split .. split + kept], split, .last_row));
        const want = try step(a, &s.pack, &serial, prompt[split + kept .. split + kept + 1], split + kept, .last_row);
        defer a.free(want);

        var cache = try test_cache.TestCache.init(a, visits);
        defer cache.deinit();
        a.free(try step(a, &s.pack, &cache, prompt[0..split], 0, .last_row));
        var g = ops.Graph.init(a, m.base.s);
        defer g.deinit();
        const opens = try a.alloc(kernels.Open, visits);
        defer a.free(opens);
        @memset(opens, .{ .g = &g });
        var sp: plugin.Spec = .{ .visits = &.{}, .keep = 4 };
        defer {
            m.base.weights.spec = null;
            for ([_]mlx.mlx_array{ sp.ring, sp.held }) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
        }
        m.base.weights.spec = &sp;
        sp.opens = opens;
        a.free(try step(a, &s.pack, &cache, prompt[split .. split + width], split, .all_rows));
        sp.opens = null;
        // Nothing of the block reached the caches.
        try testing.expectEqual(@as(u64, split), test_cache.ops.len(cache.handle(), 0));
        var h = [_]abi.Handle{cache.handle()};
        const pos = [_]u32{split};
        const r: run_mod.Run = .{ .mode = .verify, .rows = @intCast(kept), .positions = &pos, .caches = &h, .cache_ops = &test_cache.ops };
        var c = r.ctx(&g, &m.base.shared);
        try kernels.commit(&c, &m.base.weights.body.cfg, opens, @intCast(kept));
        try testing.expectEqual(@as(u64, split + kept), test_cache.ops.len(cache.handle(), 0));
        const got = try step(a, &s.pack, &cache, prompt[split + kept .. split + kept + 1], split + kept, .last_row);
        defer a.free(got);
        try testing.expect(parity.qwen4CompareRows(got, want, 1, VOCAB).min_cos > 0.999);
    }
}

/// One batched decode step: slot i's `tokens[i]` at `positions[i]` on `caches[i]`; the rows' logits.
fn batchedStep(a: std.mem.Allocator, pack: *conformance.Loaded, caches: []test_cache.TestCache, tokens: []const u32, positions: []const u32) ![]f32 {
    var handles: [8]abi.Handle = undefined;
    var slots: [8]u32 = undefined;
    for (caches, 0..) |*c, i| {
        handles[i] = c.handle();
        slots[i] = @intCast(i);
    }
    var out: abi.ForwardOut = .{};
    const v = pack.table.forward(pack.model, &.{ .mode = .batched_rows, .logits = .last_row, .rows = 1, .tokens = .of(tokens), .slots = .of(slots[0..caches.len]), .positions = .of(positions), .caches = .of(handles[0..caches.len]), .cache_ops = &test_cache.ops }, &out);
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

// Bar: slots of different lengths step twice as batched forwards (the second reads the caches and conv states the
// first left per slot). CPU, a slot at a time: each row is what its slot serves alone, to the rows' bar. GPU, rows of
// the kernels: a slot's row does not move with its place or its neighbours, to 1e-5 of the top logit.
test "inkling2 proof: batched slots serve what each slot serves alone" {
    const a = testing.allocator;
    try batchedAlone(a);
    if (!(mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable())) try batchedApart(a);
}

/// Three slots, each a prompt prefix of its own length, prefilled on fresh caches.
const Slots = struct {
    caches: [3]test_cache.TestCache = undefined,
    made: usize = 0,

    fn init(s: *Slots, a: std.mem.Allocator, pack: *conformance.Loaded, lens: [3]usize) !void {
        for (lens, 0..) |n, i| {
            s.caches[i] = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
            s.made = i + 1;
            a.free(try step(a, pack, &s.caches[i], prompt[0..n], 0, .last_row));
        }
    }

    fn deinit(s: *Slots) void {
        for (s.caches[0..s.made]) |*c| c.deinit();
    }

    /// One batched step of every slot, slot i at `lens[i] + round`.
    fn next(s: *Slots, a: std.mem.Allocator, pack: *conformance.Loaded, lens: [3]usize, round: usize) ![]f32 {
        var tokens: [3]u32 = undefined;
        var positions: [3]u32 = undefined;
        for (lens, 0..) |n, i| {
            tokens[i] = prompt[n + round];
            positions[i] = @intCast(n + round);
        }
        return batchedStep(a, pack, &s.caches, &tokens, &positions);
    }
};

fn batchedAlone(a: std.mem.Allocator) !void {
    const lens = [_]usize{ 5, 9, 14 };
    var s = try Setup.init(a, false);
    defer s.deinit(a);
    var alone: Slots = .{};
    defer alone.deinit();
    var slots: Slots = .{};
    defer slots.deinit();
    try alone.init(a, &s.pack, lens);
    try slots.init(a, &s.pack, lens);
    for (0..2) |round| {
        var want: [lens.len * VOCAB]f32 = undefined;
        for (lens, 0..) |n, i| {
            const row = try step(a, &s.pack, &alone.caches[i], prompt[n + round ..][0..1], n + round, .last_row);
            defer a.free(row);
            @memcpy(want[i * VOCAB ..][0..VOCAB], row);
        }
        const got = try slots.next(a, &s.pack, lens, round);
        defer a.free(got);
        try expectRows(got, &want);
    }
}

fn batchedApart(a: std.mem.Allocator) !void {
    // A length is a sequence: the first run's slots, the same in another order, and its first slot beside two others.
    const runs = [_][3]usize{ .{ 5, 9, 14 }, .{ 14, 5, 9 }, .{ 7, 11, 5 } };
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    var slots: [runs.len]Slots = @splat(.{});
    defer for (&slots) |*x| x.deinit();
    for (&slots, runs) |*x, lens| try x.init(a, &s.pack, lens);
    for (0..2) |round| {
        var rows: [runs.len][]f32 = undefined;
        var made: usize = 0;
        defer for (rows[0..made]) |r| a.free(r);
        for (&slots, runs, 0..) |*x, lens, r| {
            rows[r] = try x.next(a, &s.pack, lens, round);
            made = r + 1;
        }
        for (runs[1..], rows[1..]) |lens, got| for (lens, 0..) |n, i| {
            const j = std.mem.indexOfScalar(usize, &runs[0], n) orelse continue;
            try expectSameRow(got[i * VOCAB ..][0..VOCAB], rows[0][j * VOCAB ..][0..VOCAB]);
        };
    }
}

fn expectSameRow(got: []const f32, want: []const f32) !void {
    var top: f32 = 0;
    var d: f32 = 0;
    for (got, want) |x, y| {
        top = @max(top, @abs(y));
        d = @max(d, @abs(x - y));
    }
    if (!(d <= 1e-5 * top)) std.debug.print("[inkling2 proof] batched row moved {e} of top {d:.4}\n", .{ d, top });
    try testing.expect(d <= 1e-5 * top);
}

// Bar: a sliding visit's cache is a ring of window + slack rows, declared to the engine; a ring short enough to drop
// rows serves, through the engine's own cache and past its first grow, the full buffer's logits bit for bit.
test "inkling2 proof: a sliding ring that drops rows serves what the full buffer serves" {
    const a = testing.allocator;
    const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
    const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
    const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
    const run_mod = @import("../../sdk/bases/run.zig");
    const sliding_ring = @import("../../sdk/config/sliding_ring.zig");
    for (streams()) |gpu| {
        var s = try Setup.init(a, gpu);
        defer s.deinit(a);
        const m: *plugin.Loaded = @ptrCast(@alignCast(s.pack.model));
        const cfg = &m.base.weights.body.cfg;
        try testing.expectEqual(sliding_ring.rows(cfg.sliding_window), cfg.ring);
        try testing.expectEqual(cfg.ring, s.pack.parsed.cache.sliding_ring);
        var full = m.base.weights;
        full.body.cfg.ring = 0;
        var ringed = m.base.weights;
        ringed.body.cfg.ring = cfg.sliding_window + 2;
        const visits = s.pack.parsed.shell.num_hidden_layers;
        var kvs: [2]KVCache = undefined;
        var entries: [2][]SSMCacheEntry = undefined;
        var slots: [2]cache_ops.Slot = undefined;
        var made: usize = 0;
        defer for (kvs[0..made], entries[0..made]) |*c, es| {
            c.deinit();
            for (es) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            a.free(es);
        };
        for (&kvs, &entries, &slots) |*c, *es, *sl| {
            c.* = try KVCache.init(a, visits);
            es.* = try a.alloc(SSMCacheEntry, visits);
            for (es.*) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            made += 1;
            sl.* = .{ .kv = c, .ssm = es.* };
        }
        // A prompt, a step, a chunk past the cache's first capacity, steps, a chunk past the next (a grow that drops
        // ring rows, in whole blocks), steps.
        var ids: [620]i32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast((i * 37 + 11) % VOCAB);
        const steps = [_][2]usize{ .{ 0, 10 }, .{ 10, 11 }, .{ 11, 311 }, .{ 311, 312 }, .{ 312, 313 }, .{ 313, 613 }, .{ 613, 614 }, .{ 614, 615 } };
        for (steps) |st| {
            var g = ops.Graph.init(a, m.base.s);
            defer g.deinit();
            const n: c_int = @intCast(st[1] - st[0]);
            const tok = try g.fromSlice(i32, ids[st[0]..st[1]], &.{ 1, n });
            var outs: [2]ops.Array = undefined;
            for ([_]*const plugin.Model.Weights{ &full, &ringed }, &slots, 0..) |w, *sl, arm| {
                const h = [_]abi.Handle{sl.handle()};
                const pos = [_]u32{@intCast(st[0])};
                const r: run_mod.Run = .{ .mode = if (n == 1) .decode else .prefill, .rows = n, .positions = &pos, .caches = &h, .cache_ops = &cache_ops.ops, .logits = .all_rows };
                outs[arm] = (try plugin.Model.forward(w, &m.base.shared, &g, tok, r)).logits;
            }
            try ops.expectBitEqual(outs[1], outs[0]);
        }
        for (kvs, 0..) |c, arm| {
            var dropped = false;
            for (c.entries) |e| dropped = dropped or e.base > 0;
            try testing.expectEqual(arm == 1, dropped);
        }
    }
}

// Bar: the batched slots' one-launch read against the per-slot read it replaces (a mask and MLX's attention a slot),
// at the same slots: 2, 8 and 10 (two launches) slots of their own lengths, each a view of the first keys of a longer
// buffer as a cache holds them, on a sliding and a global layer, at the fixture's 16-wide heads and the model's 128.
// f32 1e-4, bf16 2e-2 of unit-scale values.
test "inkling2 kernels: batched slots read in one launch as the per-slot read does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const Geo = struct { heads: c_int, kv_heads: c_int, hd: c_int, extent: c_int, window: c_int };
    var seed: u64 = 16000;
    for ([_]Geo{ .{ .heads = 4, .kv_heads = 2, .hd = 16, .extent = 8, .window = 6 }, .{ .heads = 24, .kv_heads = 8, .hd = 128, .extent = 1024, .window = 1024 } }) |geo| for ([_]usize{ 2, 8, 10 }) |n| for ([_]bool{ true, false }) |sliding| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const slots: c_int = @intCast(n);
        var ks: [10]ops.Array = undefined;
        var vs: [10]ops.Array = undefined;
        for (0..n) |i| {
            const keys: c_int = @intCast(3 + 157 * i);
            const kv = [2]ops.Array{ try randomRow(&g, a, seed + 2 * i, &.{ 1, geo.kv_heads, keys + 24, geo.hd }, 0.5, dt), try randomRow(&g, a, seed + 2 * i + 1, &.{ 1, geo.kv_heads, keys + 24, geo.hd }, 1.0, dt) };
            ks[i] = try g.sliceAxis(kv[0], 2, 0, keys);
            vs[i] = try g.sliceAxis(kv[1], 2, 0, keys);
        }
        const q = try randomRow(&g, a, seed + 100, &.{ 1, slots, geo.heads, geo.hd }, 0.5, dt);
        const bias = try randomRow(&g, a, seed + 101, &.{ 1, slots, geo.heads, geo.extent }, 1.0, dt);
        const sinks = try randomRow(&g, a, seed + 102, &.{ 1, slots, geo.heads }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 103, &.{slots * geo.heads * geo.hd}, 2.0, .float32));
        const window: c_int = if (sliding) geo.window else 0;
        const want = try kernels.slotsChain(&g, try g.transpose(q, &.{ 0, 2, 1, 3 }), ks[0..n], vs[0..n], bias, sinks, window, gate);
        const got = try kernels.slotsRead(&g, q, ks[0..n], vs[0..n], bias, try g.contiguous(try g.astype(sinks, .float32)), window, gate);
        try expectClose(got, want, if (dt == .float32) 1e-4 else 2e-2, "batched slots' read");
    };
}

const ops = @import("../../mlx/ops.zig");
const layers = @import("layers.zig");
const kernels = @import("kernels.zig");
fn randomRow(g: *ops.Graph, a: std.mem.Allocator, seed: u64, shape: []const c_int, scale: f32, dt: ops.Dtype) !ops.Array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try fixture.randomF32(a, seed, n, scale);
    defer a.free(data);
    return g.astype(try g.fromSlice(f32, data, shape), dt);
}

fn expectClose(got: ops.Array, want: ops.Array, tol: f32, what: []const u8) !void {
    try testing.expectEqual(mlx.mlx_array_size(want), mlx.mlx_array_size(got));
    const d = parity.maxAbsDiffF32(got, want, fx_stream) catch |e| {
        std.debug.print("[inkling2 kernels] {s}: {s}\n", .{ what, @errorName(e) });
        return e;
    };
    if (!(d <= tol)) std.debug.print("[inkling2 kernels] {s}: max diff {d:.6} over {d:.6}\n", .{ what, d, tol });
    try testing.expect(d <= tol);
}

var fx_stream: mlx.mlx_stream = undefined;

/// The fixture pack on the GPU's stream, for a stage's kernels against its op chain.
const Stage = struct {
    arena: std.heap.ArenaAllocator,
    pack: conformance.Loaded,
    body: *const layers.Body,

    fn init(a: std.mem.Allocator) !Stage {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena.allocator()), .{ .gpu = true });
        const m: *plugin.Loaded = @ptrCast(@alignCast(pack.model));
        fx_stream = m.base.s;
        return .{ .arena = arena, .pack = pack, .body = &m.base.weights.body };
    }

    fn deinit(s: *Stage) void {
        s.pack.close();
        s.arena.deinit();
    }
};

// Layer 0 opens the first block and reads its own input; 1 is global inside it; 2 opens one in the loop; 3 rides it
// and 5 is the loop's global layer.
const StageCase = struct { layer: usize, blocks: c_int, messages: bool };
const stage_cases = [_]StageCase{
    .{ .layer = 0, .blocks = 0, .messages = false }, .{ .layer = 1, .blocks = 1, .messages = false },
    .{ .layer = 2, .blocks = 1, .messages = false }, .{ .layer = 2, .blocks = 1, .messages = true },
    .{ .layer = 3, .blocks = 2, .messages = true },  .{ .layer = 5, .blocks = 3, .messages = true },
};

// Bar: each stage's kernels against the op chain they stand for, on the same inputs, at one row and at a prompt's
// rows. f32: 1e-4, and 2e-2 where the chain goes through the GPU's f32 matmul (good to about 1e-3 per product). bf16:
// a bf16 step of the largest values (0.05 to 0.1), since a last-bit difference rounds an element the other way.
test "inkling2 kernels: the stream kernels serve the projections of the chain they replace" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const hidden: c_int = @intCast(s.body.cfg.hidden_size);
    var seed: u64 = 7000;
    for (stage_cases) |case| for ([_]c_int{ 1, 7 }) |rows| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &s.body.layers[case.layer];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const h = try randomRow(&g, a, seed, &.{ 1, rows, hidden }, 1.0, dt);
        const blocks: ?ops.Array = if (case.blocks > 0) try randomRow(&g, a, seed + 1, &.{ 1, rows, case.blocks, hidden }, 1.0, .float32) else null;
        const bank: ?[2]ops.Array = if (case.messages) .{ try randomRow(&g, a, seed + 2, &.{ 1, rows, 2, hidden }, 1.0, .float32), try randomRow(&g, a, seed + 3, &.{ 1, rows, hidden }, 1.0, .float32) } else null;
        try testing.expect(kernels.runs(&g, L, &s.body.cfg, h));
        const got = try kernels.pre(&g, L, s.body, h, blocks, bank);
        const messages: ?ops.Array = if (bank) |bk| try layers.messagesFor(&g, L, bk[0], bk[1]) else null;
        const want = try layers.project(&g, L, s.body, h, blocks, messages);
        const tol: f32 = if (dt == .float32) 2e-2 else 0.1;
        for (got.proj, want.proj) |x, y| try expectClose(x, y, tol, "projection");
        try testing.expectEqual(want.dyn != null, got.dyn != null);
        if (want.dyn) |d| for (got.dyn.?, d) |x, y| try expectClose(x, y, tol, "conv taps");
        try expectClose(got.normed, want.normed, 1e-4, "normed stream");
        // The chain's messages are `[1, S, 4, D]`.
        if (messages) |m| try expectClose(got.messages.?, try g.contiguous(try g.transpose(m, &.{ 2, 1, 0, 3 })), 2e-2, "messages");
    };
    var g = ops.Graph.init(a, fx_stream);
    defer g.deinit();
    const h = try randomRow(&g, a, seed + 20, &.{ 1, 3, hidden }, 1.0, .bfloat16);
    const blocks = try randomRow(&g, a, seed + 21, &.{ 1, 3, 3, hidden }, 1.0, .float32);
    const nw = try randomRow(&g, a, seed + 22, &.{hidden}, 1.0, .float32);
    const got = (try kernels.finalStream(&g, s.body, h, blocks, nw)) orelse return error.TestUnexpectedResult;
    try expectClose(got, try g.astype(try g.rmsNorm(try layers.finalInput(&g, s.body, h, blocks), nw, s.body.cfg.rms_norm_eps), .bfloat16), 0.05, "final stream");
}

/// Random projections of `rows` rows for a layer, as `layers.project` hands them over.
fn randomProjected(g: *ops.Graph, a: std.mem.Allocator, seed: u64, L: *const layers.Layer, cfg: *const @TypeOf(@as(layers.Body, undefined).cfg), rows: c_int, dt: ops.Dtype) !layers.Projected {
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const kv_dim: c_int = @intCast(cfg.kvDim());
    const taps: c_int = @intCast(cfg.num_key_value_heads * cfg.sconv_kernel);
    var proj: layers.Projected = undefined;
    for (&proj.proj, [_]c_int{ heads * hd, heads * hd, heads * @as(c_int, @intCast(cfg.d_rel)), heads, kv_dim, kv_dim }, 0..) |*x, n, i|
        x.* = try randomRow(g, a, seed + i, &.{ 1, rows, n }, 1.5, dt);
    proj.dyn = if (L.global) .{ try randomRow(g, a, seed + 50, &.{ 1, rows, taps }, 0.3, dt), try randomRow(g, a, seed + 51, &.{ 1, rows, taps }, 0.3, dt) } else null;
    proj.stream = proj.proj[0];
    proj.normed = proj.proj[0];
    return proj;
}

// Bar: as above, on a sliding and a global layer. One row: with and without log-scaling, its bias row for a view
// shorter than, as long as and longer than the relative extent. A prompt's rows: the clipped relative states.
test "inkling2 kernels: the prep kernel serves the values of the chain it replaces" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const body = s.body;
    const cfg = &body.cfg;
    const taps: c_int = @intCast(cfg.sconv_kernel);
    const width: c_int = @intCast(2 * cfg.kvDim() + 2 * cfg.hidden_size);
    var seed: u64 = 900;
    for ([_]usize{ 0, 1 }) |li| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &body.layers[li];
        const extent = ops.shape(L.rel_proj)[1];
        const tol: f32 = if (dt == .float32) 1e-4 else 0.05;
        for ([_]f32{ 1.0, 1.25 }) |tau| for ([_]c_int{ 3, 0, 5 }) |past| {
            const keys = if (past == 3) 3 else extent + past;
            var g = ops.Graph.init(a, fx_stream);
            defer g.deinit();
            seed += 100;
            const proj = try randomProjected(&g, a, seed, L, cfg, 1, dt);
            const st = try randomRow(&g, a, seed + 60, &.{ 1, taps - 1, width }, 1.0, .float32);
            const got = (try kernels.prep(&g, L, cfg, &proj.proj, proj.dyn, st, tau, keys)) orelse return error.TestUnexpectedResult;
            const want = try layers.prepare(&g, L, body, &proj, st);
            try expectClose(got.q, try g.astype(try g.mulScalar(try g.astype(want.q, .float32), tau), dt), tol, "q");
            try expectClose(got.k, want.k, tol, "k");
            try expectClose(got.v, want.v, tol, "v");
            try expectClose(got.gate, want.gate, tol, "gate");
            // The chain's one-row bias: the logits' tail, zero past the extent.
            var bias = if (keys < extent) try g.sliceAxis(want.rel, 1, extent - keys, extent) else want.rel;
            if (keys > extent) bias = try g.padAxis(bias, 1, keys - extent, 0);
            try expectClose(got.bias, try g.astype(try g.mulScalar(bias, tau), dt), @max(tol, 2e-2), "bias");
        };
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 100;
        const proj = try randomProjected(&g, a, seed, L, cfg, 7, dt);
        const st = try randomRow(&g, a, seed + 60, &.{ 1, taps - 1, width }, 1.0, .float32);
        const got = (try kernels.prep(&g, L, cfg, &proj.proj, proj.dyn, st, 1.0, 0)) orelse return error.TestUnexpectedResult;
        const want = try layers.prepare(&g, L, body, &proj, st);
        try expectClose(got.q, want.q, tol, "rows' q");
        // The chain's k and v are transposed views.
        try expectClose(got.k, try g.contiguous(want.k), tol, "rows' k");
        try expectClose(got.v, try g.contiguous(want.v), tol, "rows' v");
        try expectClose(got.gate, want.gate, tol, "rows' gate");
        try expectClose(got.rel, want.rel, tol, "rows' relative states");
    };
}

// Bar: as above; the picks are the chain's experts, each at its chain weight.
test "inkling2 kernels: the mid kernels serve the values of the chain they replace" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const body = s.body;
    const cfg = &body.cfg;
    const hidden: c_int = @intCast(cfg.hidden_size);
    const taps: c_int = @intCast(cfg.sconv_kernel);
    const top_k: usize = cfg.top_k;
    const q_dim: c_int = @intCast(cfg.num_attention_heads * cfg.head_dim);
    var seed: u64 = 5000;
    for (stage_cases) |case| for ([_]c_int{ 1, 7 }) |rows| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &body.layers[case.layer];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const attn = try randomRow(&g, a, seed, &.{ 1, rows, q_dim }, 1.0, dt);
        const h = try randomRow(&g, a, seed + 1, &.{ 1, rows, hidden }, 1.0, dt);
        const normed = try randomRow(&g, a, seed + 2, &.{ 1, rows, hidden }, 1.0, .float32);
        const blocks: ?ops.Array = if (case.blocks > 0) try randomRow(&g, a, seed + 3, &.{ 1, rows, case.blocks, hidden }, 1.0, .float32) else null;
        const messages: ?ops.Array = if (case.messages) try randomRow(&g, a, seed + 4, &.{ 4, rows, hidden }, 1.0, .float32) else null;
        const st = try randomRow(&g, a, seed + 5, &.{ 1, taps - 1, 2 * @as(c_int, @intCast(cfg.kvDim())) + 2 * hidden }, 1.0, .float32);
        const got = try kernels.mid(&g, L, body, attn, null, h, normed, blocks, st, messages);
        // The chain's messages are `[1, S, 4, D]`.
        const want = try layers.mid(&g, L, body, attn, try g.astype(h, .float32), normed, blocks, st, if (messages) |m| try g.contiguous(try g.transpose(try g.reshape(m, &.{ 4, rows, 1, hidden }), &.{ 2, 1, 0, 3 })) else null);
        const tol: f32 = if (dt == .float32) 1e-4 else 0.1;
        try expectClose(got.residual, want.residual, tol, "residual");
        try expectClose(got.x, want.x, @max(tol, 2e-2), "mlp input");
        try testing.expectEqual(want.blocks != null, got.blocks != null);
        if (want.blocks) |b| try expectClose(got.blocks.?, b, 1e-6, "block stack");
        const gi = try hostValues(a, &g, got.idx);
        defer a.free(gi);
        const wi = try hostValues(a, &g, want.idx);
        defer a.free(wi);
        const gw = try hostValues(a, &g, got.weights);
        defer a.free(gw);
        const ww = try hostValues(a, &g, want.weights);
        defer a.free(ww);
        for (gi, gw, 0..) |e, weight, at| {
            const row = at / top_k * top_k;
            const pick = std.mem.indexOfScalar(f32, wi[row .. row + top_k], e) orelse {
                std.debug.print("[inkling2 kernels] layer {d}: expert {d} is not a chain pick {any}\n", .{ case.layer, e, wi[row .. row + top_k] });
                return error.TestUnexpectedResult;
            };
            try testing.expectApproxEqAbs(ww[row + pick], weight, 2e-2 * cfg.route_scale);
        }
    };
}

fn hostValues(a: std.mem.Allocator, g: *ops.Graph, x: ops.Array) ![]f32 {
    return conformance.consumeF32(a, .{ .ctx = g.escape(try g.astype(try g.contiguous(x), .float32)).ctx });
}

const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const device = @import("../../mlx/device.zig");

/// A random expert bank `shape` (`[E, out, in]`) quantized by MLX into a packed layout.
fn packedBank(g: *ops.Graph, a: std.mem.Allocator, seed: u64, shape: []const c_int, bits: u32, group: u32, mode: QuantMode) !@TypeOf(@as(layers.Layer, undefined).gate_up) {
    const w = try randomRow(g, a, seed, shape, 0.1, .float32);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, w, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{}, g.s));
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&q, parts, 0));
    const words = try g.adopt(q);
    var sc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&sc, parts, 1));
    var bank: @TypeOf(@as(layers.Layer, undefined).gate_up) = .{ .w = words, .s = try g.adopt(sc), .qp = .{ .bits = bits, .group_size = group, .mode = mode } };
    if (mode == .affine) {
        var bi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&bi, parts, 2));
        bank.b = try g.adopt(bi);
    }
    return bank;
}

/// The affine widths an expert bank may be re-quantized to (`tests/convert_inkling2.py --experts`).
const affine_bits = [_]u32{ 2, 3, 4, 5, 6, 8 };
const affine_groups = [_]u32{ 32, 64, 128 };
// Bar: one row's fused output projection serves `projectRows(gated(read, gate))` within one bf16 rounding, on every
// affine width and group a pack's attention may carry and on a dense weight.
test "inkling2 kernels: the output projection kernel serves the gated read through the projection" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const n: c_int = 256;
    const heads: c_int = 24;
    const hd: c_int = 128;
    var seed: u64 = 9100;
    for ([_]?u32{ null, 2, 3, 4, 5, 6, 8 }) |bits| for (affine_groups) |group| {
        if (bits == null and group != 64) continue;
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        var o: layers.Linear = .{ .w = try randomRow(&g, a, seed, &.{ n, heads * hd }, 0.02, .bfloat16) };
        if (bits) |b| {
            const q = try packedBank(&g, a, seed, &.{ 1, n, heads * hd }, b, group, .affine);
            o = .{ .w = try g.reshape(q.w, &.{ n, -1 }), .s = try g.reshape(q.s, &.{ n, -1 }), .b = try g.reshape(q.b, &.{ n, -1 }), .bits = b, .group_size = group };
        }
        try testing.expect(kernels.oFormat(&o, .bfloat16) != null);
        const att = try randomRow(&g, a, seed + 1, &.{ 1, heads, 1, hd }, 1.0, .bfloat16);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 2, &.{heads * hd}, 2.0, .float32));
        const want = try layers.projectRows(&g, &o, try kernels.gated(&g, att, gate, 1));
        try expectClose(try kernels.oGated(&g, &o, att, gate), want, 4e-2, "output projection");
    };
}

const ExpertFormat = struct { mode: ?QuantMode, bits: u32 = 0, group: u32 = 0 };
/// Each width at the fixture's 64-wide stream: two scale groups of 32.
const affine_formats = blk: {
    var all: [affine_bits.len]ExpertFormat = undefined;
    for (affine_bits, 0..) |bits, i| all[i] = .{ .mode = .affine, .bits = bits, .group = 32 };
    break :blk all;
};

// Bar: the tail's launches against `layers.tail` from the router's own picks, with the new conv state (the old rows
// behind the rows' k, v and output projections and the MLP's output): on the fixture's dense bank and on each packed
// and affine layout a bank can load in, with and without per-expert scales, at one row (the packed banks read in place), a few
// rows (unsorted pairs) and a prompt's (sorted). f32 2e-3 (MLX's fp kernels sum in another order), bf16 0.3 (two steps of its largest sums).
test "inkling2 kernels: the expert tail kernels serve the stream and the conv state of the chain they replace" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    var body = s.body.*;
    const hidden: c_int = @intCast(body.cfg.hidden_size);
    const experts: c_int = @intCast(body.cfg.num_experts);
    const kv_dim: c_int = @intCast(body.cfg.kvDim());
    const taps: c_int = @intCast(body.cfg.sconv_kernel);
    var seed: u64 = 9000;
    for ([_]ExpertFormat{ .{ .mode = null }, .{ .mode = .nvfp4, .bits = 4, .group = 16 }, .{ .mode = .mxfp4, .bits = 4, .group = 32 }, .{ .mode = .mxfp8, .bits = 8, .group = 32 } } ++ affine_formats) |f| for ([_]c_int{ 1, 7, 43 }) |rows| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| for ([_]bool{ true, false }) |scaled| {
        // MLX's fp matmuls take bf16 rows; one row reads a packed bank in place at either.
        if (f.mode != null and dt == .float32 and rows > 1) continue;
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        var L = body.layers[3];
        body.cfg.moe_intermediate_size = s.body.cfg.moe_intermediate_size;
        if (f.mode) |mode| {
            // Two scale groups wide: MLX's sorted MXFP8 gather is not finite on a bank one group (32) wide.
            body.cfg.moe_intermediate_size = 64;
            L.gate_up = try packedBank(&g, a, seed, &.{ experts, 128, hidden }, f.bits, f.group, mode);
            L.down = try packedBank(&g, a, seed + 1, &.{ experts, hidden, 64 }, f.bits, f.group, mode);
        }
        if (scaled) {
            L.gate_up_scale = try g.addScalar(try randomRow(&g, a, seed + 2, &.{experts}, 0.3, .float32), 1.5);
            L.down_scale = try g.addScalar(try randomRow(&g, a, seed + 3, &.{experts}, 0.3, .float32), 1.5);
        }
        const cfg = &body.cfg;
        const x = try randomRow(&g, a, seed + 4, &.{ 1, rows, hidden }, 1.0, dt);
        const r = try layers.route(&g, &L, cfg, x);
        const residual = try randomRow(&g, a, seed + 5, &.{ 1, rows, hidden }, 1.0, .float32);
        const st = try randomRow(&g, a, seed + 6, &.{ 1, taps - 1, 2 * kv_dim + 2 * hidden }, 1.0, .float32);
        const kp = try randomRow(&g, a, seed + 7, &.{ 1, rows, kv_dim }, 1.0, dt);
        const vp = try randomRow(&g, a, seed + 8, &.{ 1, rows, kv_dim }, 1.0, dt);
        const op = try randomRow(&g, a, seed + 9, &.{ 1, rows, hidden }, 1.0, dt);
        const got = try kernels.experts(&g, &L, cfg, &.{ .x = x, .idx = r.idx, .weights = r.weights, .residual = residual, .blocks = null, .op = op }, kp, vp, st, null);
        const want = try layers.tail(&g, &L, &body, &.{ .x = x, .idx = r.idx, .weights = r.weights, .residual = residual, .blocks = null, .attn_state = x }, st);
        // The bf16 sums reach 20 and over, where one bf16 step is 0.125: a last-bit flip or two is not a miss.
        const tol: f32 = if (dt == .float32) 2e-3 else 0.3;
        var label: [96]u8 = undefined;
        try expectClose(got.h, want.h, tol, try std.fmt.bufPrint(&label, "tail stream ({s}, {d} rows, {s}, scaled {})", .{ if (f.mode) |m| @tagName(m) else "dense", rows, @tagName(dt), scaled }));
        // The k, v and attention windows: the last rows of the old state behind the rows' projections.
        const fed = try g.concat(&.{ try g.sliceAxis(st, 2, 0, 2 * kv_dim + hidden), try g.astype(try g.concat(&.{ kp, vp, op }, 2), .float32) }, 1);
        const left = try g.sliceAxis(fed, 1, rows, rows + taps - 1);
        try expectClose(got.state, try g.concat(&.{ left, want.mlp_state }, 2), tol, "conv state");
    };
}

// Bar: the packed kernels take an affine bank at every width a pack may be re-quantized to, read in place.
test "inkling2 kernels: an affine bank is a packed format at every width" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    var seed: u64 = 14000;
    for (affine_bits) |bits| for (affine_groups) |group| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const bank = try packedBank(&g, a, seed, &.{ 4, 64, 256 }, bits, group, .affine);
        try testing.expect(kernels.packedFormat(bank, 256) != null);
    };
}

// Bar: an affine expert bank at every width a pack may be re-quantized to (2-6 and 8 bits, groups 32, 64 and 128)
// gathers as the dequantized bank does, unsorted (a row's picks) and sorted (a prompt's), f32 rows: 1e-3 of unit RMS.
test "inkling2 experts: an affine bank gathers at every width as its dequantized rows do" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    var seed: u64 = 12000;
    for (affine_bits) |bits| for (affine_groups) |group| for ([_]bool{ false, true }) |sorted| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const in: c_int = @intCast(4 * group);
        const bank = try packedBank(&g, a, seed, &.{ 8, 64, in }, bits, group, .affine);
        const x = try randomRow(&g, a, seed + 1, &.{ 6, 1, in }, 1.0, .float32);
        const picks = try g.fromSlice(u32, &.{ 5, 5, 0, 3, 3, 7 }, &.{6});
        const got = try layers.gather(&g, x, &bank, picks, sorted);
        var dq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_dequantize(&dq, bank.w, bank.s, bank.b, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, g.s));
        const dense = try g.adopt(dq);
        const want = try g.matmul(x, try g.transpose(try g.take(dense, picks, 0), &.{ 0, 2, 1 }));
        var label: [64]u8 = undefined;
        try expectClose(got, want, 1e-3, try std.fmt.bufPrint(&label, "affine {d} bits, group {d}, sorted {}", .{ bits, group, sorted }));
    };
}

// Bar: only an affine bank on a GPU before Apple9 (M1 g13, M2 g14), which has no bf16 ALU, reads bf16 rows as fp16.
test "inkling2 experts: bf16 rows go into the gather as fp16 for an affine bank on a GPU without bf16 only" {
    try testing.expectEqual(ops.Dtype.float16, layers.gatherRowsDtype(.bfloat16, .affine, true, 13));
    try testing.expectEqual(ops.Dtype.float16, layers.gatherRowsDtype(.bfloat16, .affine, true, 14));
    for ([_]u32{ 15, 16, 17, 18 }) |gen| try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, .affine, true, gen));
    // The release's banks (raw outputs past fp16), an unknown generation, a CPU stream and every other dtype stay.
    for ([_]QuantMode{ .nvfp4, .mxfp4, .mxfp8 }) |mode| try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, mode, true, 13));
    try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, .affine, true, 0));
    try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, .affine, false, 13));
    try testing.expectEqual(ops.Dtype.float32, layers.gatherRowsDtype(.float32, .affine, true, 13));
    try testing.expectEqual(ops.Dtype.float16, layers.gatherRowsDtype(.float16, .affine, true, 13));
}

// Bar: a bf16 gather served as fp16 (device generation pinned to an M1's) leaves bf16 and agrees with the plain bf16
// gather to a bf16 step, (an NVFP4 bank, which keeps bf16 rows, as a control), sorted and not.
test "inkling2 experts: the fp16 gather of a bf16 call agrees with the bf16 one" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const saved = device.device_gen_override;
    defer device.device_gen_override = saved;
    var seed: u64 = 13000;
    for ([_]ExpertFormat{ .{ .mode = .affine, .bits = 4, .group = 64 }, .{ .mode = .affine, .bits = 3, .group = 64 }, .{ .mode = .nvfp4, .bits = 4, .group = 16 } }) |f| for ([_]bool{ false, true }) |sorted| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const bank = try packedBank(&g, a, seed, &.{ 8, 64, 256 }, f.bits, f.group, f.mode.?);
        const x = try randomRow(&g, a, seed + 1, &.{ 6, 1, 256 }, 1.0, .bfloat16);
        const picks = try g.fromSlice(u32, &.{ 5, 5, 0, 3, 3, 7 }, &.{6});
        device.device_gen_override = .{ .gen = 17, .phone = false };
        const plain = try layers.gather(&g, x, &bank, picks, sorted);
        device.device_gen_override = .{ .gen = 13, .phone = false };
        const served = try layers.gather(&g, x, &bank, picks, sorted);
        try testing.expectEqual(ops.Dtype.bfloat16, ops.dtype(served));
        var label: [64]u8 = undefined;
        try expectClose(try g.astype(served, .float32), try g.astype(plain, .float32), 0.1, try std.fmt.bufPrint(&label, "{s} {d} bits, sorted {}", .{ @tagName(f.mode.?), f.bits, sorted }));
    };
}

// Bar: the flash kernel (q, k, v as halves, an online softmax from the sink) against the masked SDPA chain on the
// same rows, on a sliding and a global layer, from the sequence's start and behind earlier keys (past the window, the
// extent and the log-scaling floor), for a row count off the kernel's tiles: f32 1e-2, bf16 5e-2 of unit-RMS values.
test "inkling2 kernels: the relative attention kernel reads a prompt's rows as the masked chain does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const cfg = &s.body.cfg;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    var seed: u64 = 12000;
    for ([_]usize{ 0, 1 }) |li| for ([_]c_int{ 5, 43 }) |rows| for ([_]c_int{ 0, 3, 50 }) |offset| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &s.body.layers[li];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const total = offset + rows;
        const keys = if (L.global) total else @min(total, @as(c_int, @intCast(cfg.sliding_window)) + rows - 1);
        const q = try randomRow(&g, a, seed, &.{ 1, rows, heads, hd }, 0.6, dt);
        const p: layers.Pre = .{
            .q = q,
            .k = q,
            .v = q,
            .rel = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, @intCast(cfg.d_rel) }, 0.5, dt),
            .gate = q,
            .sinks = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads }, 1.0, dt),
            .k_state = q,
            .v_state = q,
            .stream = q,
            .normed = q,
        };
        const k = try randomRow(&g, a, seed + 3, &.{ 1, kv_heads, keys, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, keys, hd }, 1.0, dt);
        // The chain's rows are a transposed view: compared as the dense array `attend` makes of them.
        // Both arms: the flash kernel, and under `small_rows` the mask kernel with MLX's attention.
        const want = try g.contiguous(try layers.wideRead(&g, L, cfg, &p, k, v, offset));
        for ([_]bool{ true, false }) |flash| {
            try expectClose(try g.contiguous(try kernels.relative(&g, L, cfg, p.q, p.rel, p.sinks, k, v, offset, flash)), want, if (dt == .float32) 1e-2 else 5e-2, "relative read");
        }
    };
}

// Bar: the split read (keys split over threadgroups, combined under the gate) against the masked chain times its
// gate, on a sliding and a global layer, from the start and behind earlier keys (past the window, the extent and the
// log-scaling floor), at the fewest rows it takes, an odd count and the most: 5e-2 of unit-RMS values in half dtypes.
test "inkling2 kernels: the split read is the masked chain's, gated" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const cfg = &s.body.cfg;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    var seed: u64 = 13000;
    for ([_]usize{ 0, 1 }) |li| for ([_]c_int{ kernels.split_from, 5, kernels.split_rows }) |rows| for ([_]c_int{ 0, 3, 300 }) |offset| for ([_]ops.Dtype{ .bfloat16, .float16 }) |dt| {
        const L = &s.body.layers[li];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const total = offset + rows;
        const keys = if (L.global) total else @min(total, @as(c_int, @intCast(cfg.sliding_window)) + rows - 1);
        const q = try randomRow(&g, a, seed, &.{ 1, rows, heads, hd }, 0.6, dt);
        const p: layers.Pre = .{
            .q = q,
            .k = q,
            .v = q,
            .rel = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, @intCast(cfg.d_rel) }, 0.5, dt),
            .gate = q,
            .sinks = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads }, 1.0, dt),
            .k_state = q,
            .v_state = q,
            .stream = q,
            .normed = q,
        };
        const k = try randomRow(&g, a, seed + 3, &.{ 1, kv_heads, keys, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, keys, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 5, &.{rows * heads * hd}, 2.0, .float32));
        const want = try kernels.gated(&g, try g.contiguous(try layers.wideRead(&g, L, cfg, &p, k, v, offset)), gate, rows);
        const got = (try kernels.splitRead(&g, L, cfg, p.q, p.rel, p.sinks, k, v, offset, gate)) orelse return error.TestUnexpectedResult;
        try expectClose(got, want, 5e-2, "split read");
    };
}

// Bar: the drafter's Markov chain as three launches a step drafts the op chain's tokens, each with its probability to
// 1e-2 (one bf16 ulp of the top logit moves it a few percent), from three committed tokens over random heads whose
// logits stand apart (a pick a bf16 rounding cannot flip); in both, the probability reads the soft-capped logits.
test "inkling2 kernels: the fused Markov chain drafts the op chain's tokens" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const rank = 64;
    const hidden = 128;
    const vpad = 4096;
    const vocab = 4000;
    const steps = 7;
    for ([_]ops.Dtype{ .bfloat16, .float16 }) |dt| for ([_]i32{ 3, 1777, 3999 }) |first| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const seed: u64 = 15000 + @as(u64, @intCast(first));
        var d: draft.Drafter = .{
            .cfg = .{ .layers = 0, .heads = 1, .kv_heads = 1, .head_dim = 64, .intermediate = 1, .window = 2, .block = steps + 1, .rank = rank, .tap_count = 1 },
            .hidden = hidden,
            .s = fx_stream,
        };
        d.norm = try g.ones(&.{hidden}, dt);
        d.prev_rows = try randomRow(&g, a, seed, &.{ vpad, rank }, 0.5, dt);
        d.rnn = .{ .w = try randomRow(&g, a, seed + 1, &.{ 3 * rank, 2 * rank + 2 * hidden }, 0.05, dt) };
        d.rnn_bias = try randomRow(&g, a, seed + 2, &.{3 * rank}, 0.2, dt);
        d.fold = try randomRow(&g, a, seed + 3, &.{ rank, vpad }, 0.3, dt);
        d.fold_bias = try randomRow(&g, a, seed + 4, &.{ 1, 1, vpad }, 0.3, dt);
        const rows = try randomRow(&g, a, seed + 5, &.{ 1, steps, hidden }, 1.0, dt);
        const newest = try randomRow(&g, a, seed + 6, &.{ 1, 1, hidden }, 1.0, dt);
        const base = try g.astype(try randomRow(&g, a, seed + 7, &.{ 1, steps, vpad }, 4.0, dt), .float32);
        const t1 = try g.fromSlice(i32, &.{first}, &.{1});
        try testing.expect(kernels.markovRuns(&g, d.rnn.w, d.fold, rank));
        const want = try d.markov(&g, t1, rows, newest, base, vocab, 30, false);
        const got = try d.markov(&g, t1, rows, newest, base, vocab, 30, true);
        try expectClose(got.tokens, want.tokens, 0, "drafted tokens");
        try expectClose(got.confidence, want.confidence, 1e-2, "draft probabilities");
        // Logits far apart: their own top is all but sure, their capped top is not (the head caps at 30).
        const far = try g.mulScalar(base, 25);
        for ([_]bool{ false, true }) |fused| {
            const raw = try hostValues(a, &g, (try d.markov(&g, t1, rows, newest, far, vocab, 0, fused)).confidence);
            defer a.free(raw);
            const cap = try hostValues(a, &g, (try d.markov(&g, t1, rows, newest, far, vocab, 30, fused)).confidence);
            defer a.free(cap);
            try testing.expect(raw[0] > 0.5 and cap[0] < 0.1);
        }
    };
}

// Bar: a slot's rounds with the drafter's context keys and values carried over draft what rounds projecting the
// context afresh draft, the probabilities to 1e-5 (the CPU stream, f32): a ring that grows, slides past its keep, holds
// still (no new row) and jumps past the cache (projected afresh).
test "inkling2 draft: rounds that carry the context keys and values draft what fresh ones do" {
    const a = testing.allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const saved = fx_stream;
    fx_stream = cpu;
    defer fx_stream = saved;
    var g = ops.Graph.init(a, cpu);
    defer g.deinit();
    const H = 32;
    const HD = 16;
    const V = 64;
    const R = 8;
    const dt: ops.Dtype = .float32;
    var seed: u64 = 17000;
    const W = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, shape: []const c_int, scale: f32) !ops.Array {
            sd.* += 1;
            return randomRow(gr, al, sd.*, shape, scale, .float32);
        }
    };
    const lin = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, out: c_int, in: c_int) !@import("../../sdk/bases/blocks/linear.zig").Linear {
            return .{ .w = try W.of(gr, al, sd, &.{ out, in }, 0.2) };
        }
    };
    const one = try g.ones(&.{H}, dt);
    var layer: draft.Layer = .{
        .in_norm = one, .post_norm = one, .mix_norm = one,
        .q = try lin.of(&g, a, &seed, 2 * HD, H), .k = try lin.of(&g, a, &seed, HD, H), .v = try lin.of(&g, a, &seed, HD, H), .o = try lin.of(&g, a, &seed, H, 2 * HD),
        .k_ctx = try lin.of(&g, a, &seed, HD, H), .v_ctx = try lin.of(&g, a, &seed, HD, H),
        .q_norm = try g.ones(&.{HD}, dt), .k_norm = try g.ones(&.{HD}, dt), .k_norm_ctx = try g.ones(&.{HD}, dt),
        .gate = try lin.of(&g, a, &seed, 48, H), .up = try lin.of(&g, a, &seed, 48, H), .down = try lin.of(&g, a, &seed, H, 48),
    };
    var d: draft.Drafter = .{
        .cfg = .{ .layers = 1, .heads = 2, .kv_heads = 1, .head_dim = HD, .intermediate = 48, .window = 13, .block = 4, .rank = R, .tap_count = 2 },
        .hidden = H,
        .s = cpu,
    };
    d.layers = (&layer)[0..1];
    d.norm = one;
    d.mask = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
    d.alpha = try g.softmax(try W.of(&g, a, &seed, &.{ 2, 2, 1 }, 1.0), 1, true);
    d.prev_rows = try W.of(&g, a, &seed, &.{ V, R }, 0.5);
    d.rnn = .{ .w = try W.of(&g, a, &seed, &.{ 3 * R, 2 * R + 2 * H }, 0.1) };
    d.rnn_bias = try W.of(&g, a, &seed, &.{3 * R}, 0.1);
    d.rnn_norm = one;
    d.fold = try W.of(&g, a, &seed, &.{ R, V }, 0.3);
    d.fold_bias = try W.of(&g, a, &seed, &.{ 1, 1, V }, 0.3);
    const head = try lin.of(&g, a, &seed, V, H);
    const anchor = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
    const t1 = try g.fromSlice(i32, &.{7}, &.{1});
    const full = try W.of(&g, a, &seed, &.{ 1, 40, 2 * H }, 1.0);
    var cache: draft.ContextKv = .{};
    defer cache.release();
    // The ring's rows [lo, hi), ending at the round's position: grows, slides past 12 rows, holds, jumps.
    for ([_][2]c_int{ .{ 0, 5 }, .{ 0, 9 }, .{ 2, 14 }, .{ 6, 18 }, .{ 6, 18 }, .{ 28, 40 } }) |span| {
        const ring = try g.sliceAxis(full, 1, span[0], span[1]);
        const got = try d.draft(&g, t1, anchor, ring, span[1], &head, V - 4, 30, &cache);
        const want = try d.draft(&g, t1, anchor, ring, span[1], &head, V - 4, 30, null);
        try expectClose(got.tokens, want.tokens, 0, "drafted tokens");
        try expectClose(got.confidence, want.confidence, 1e-5, "draft probabilities");
    }
}

// Bar: the cut a round verifies drafts down to follows what a verified row costs against a plain step, as measured
// on the two machines it was tuned on (M4 Max: a 17.5 ms plain step, 3.5 ms a row, best cut 0.6; M5 Ultra: 11.3 ms,
// 1.35 ms, 0.45), to 0.02; 0.5 before a plain step and two draft counts are timed.
test "inkling2 draft: the verify cut follows the cost a row adds on each machine" {
    for ([_][3]f32{ .{ 17.5, 3.5, 0.6 }, .{ 11.3, 1.35, 0.45 } }) |m| {
        var c: plugin.RowCost = .{};
        try testing.expectEqual(@as(f32, 0.5), c.cut());
        for (0..6) |_| {
            c.plain.add(m[0]);
            for (1..8) |k| c.verify[k].add(m[0] + 5 + m[1] * @as(f32, @floatFromInt(k - 1)));
        }
        try testing.expectApproxEqAbs(m[2], c.cut(), 0.02);
    }
}

// Bar: the few-row router kernel's logits are each row's one-row matmul (exact f32; the matmul over all the rows
// at once is only good to ~1e-3), to 1e-5 of the largest.
test "inkling2 kernels: the router kernel's rows are the matmul's" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    for ([_]c_int{ 2, 7 }) |rows| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const x = try randomRow(&g, a, 71, &.{ 1, rows, 192 }, 1.0, .float32);
        const router = try randomRow(&g, a, 72, &.{ 24, 192 }, 0.2, .float32);
        var each: [7]ops.Array = undefined;
        for (each[0..@intCast(rows)], 0..) |*r, i| {
            const at: c_int = @intCast(i);
            r.* = try g.reshape(try g.matmul(try g.sliceAxis(x, 1, at, at + 1), try g.transpose(router, &.{ 1, 0 })), &.{ 1, 24 });
        }
        try expectClose(try kernels.routerRows(&g, x, router), try g.concat(each[0..@intCast(rows)], 0), 1e-5, "router rows");
    }
}

// Bar: the gate's product in f32, rounded once: the chain's bytes.
test "inkling2 kernels: the gated kernel is the attention read times its gate, rounded once" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const x = try randomRow(&g, a, 31, &.{ 1, 4, 1, 16 }, 2.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, 32, &.{64}, 2.0, .float32));
        const want = try g.astype(try g.mul(try g.astype(try g.reshape(x, &.{ 1, 1, 64 }), .float32), gate), dt);
        try expectClose(try kernels.gated(&g, x, gate, 1), want, 0, "gated rows");
    }
}
