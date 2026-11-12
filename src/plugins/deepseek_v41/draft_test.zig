//! The DSpark draft over release-shaped fixtures of random weights (`decode/fixture.zig` `initDraft`): four tensor
//! ranks on their slices of the stages and rows of the heads draft what one model drafts, through rank 0's draft and
//! the workers' `tpWorkerDraft`, issuing the same collectives in the same order; the ubench times one rank's share
//! (`DSV41_DEC_UBENCH`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const iou = @import("../../sdk/util/io_util.zig");
const decode = @import("decode/decode.zig");
const fixture = @import("decode/fixture.zig");
const dsv41 = @import("deepseek_v41.zig");
const device = @import("../../mlx/device.zig");

const A = mlx.mlx_array;
const testing = std.testing;
const ranks = 4;

fn gpuUnavailable() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu();
}

const Group = fixture.Group;

const Opts = struct { experts: u32 = 16, vocab: u32 = 4096, ctx: usize = 120, act: mlx.mlx_dtype = .float32 };

/// One whole model and four ranks of it, each at the same weights and windows.
const Models = struct {
    whole: *fixture.Fixture,
    rank: [ranks]*fixture.Fixture,

    fn init(gpa: std.mem.Allocator, o: Opts) !Models {
        const base: fixture.DraftOpts = .{ .experts = o.experts, .vocab = o.vocab, .ctx = o.ctx, .act = o.act };
        var ms: Models = undefined;
        ms.whole = try fixture.Fixture.initDraft(gpa, base);
        errdefer ms.whole.deinit();
        var made: usize = 0;
        errdefer for (ms.rank[0..made]) |f| f.deinit();
        for (&ms.rank, 0..) |*f, r| {
            var t = base;
            t.tp = .{ .rank = @intCast(r), .size = ranks };
            f.* = try fixture.Fixture.initDraft(gpa, t);
            made += 1;
        }
        return ms;
    }

    fn deinit(ms: *Models) void {
        ms.whole.deinit();
        for (ms.rank) |f| f.deinit();
    }

    fn all(ms: *Models) [ranks + 1]*fixture.Fixture {
        return .{ ms.whole, ms.rank[0], ms.rank[1], ms.rank[2], ms.rank[3] };
    }
};

fn freeDrafts(gpa: std.mem.Allocator, ds: []dsv41.DsparkDraft) void {
    for (ds) |*d| d.deinit(gpa);
}

/// What rank 0 hands the workers (`ClusterHooksState.tp_draft`).
const Sent = struct {
    var gens: [dsv41.region_rows]u64 = undefined;
    var p0s: [dsv41.region_rows]usize = undefined;
    var t1s: [dsv41.region_rows]u32 = undefined;
    var n: usize = 0;
    var sends: usize = 0;

    fn send(gens_: []const u64, p0s_: []const usize, t1s_: []const u32) anyerror!void {
        n = gens_.len;
        @memcpy(gens[0..n], gens_);
        @memcpy(p0s[0..n], p0s_);
        @memcpy(t1s[0..n], t1s_);
        sends += 1;
    }
};

/// Rank 0's drafts of `states` (`gens`) after the group settles, each worker on its own copies of them through
/// `tpWorkerDraft`, looked up by gen.
fn groupDraft(gpa: std.mem.Allocator, ms: *Models, states: [ranks][]const *dsv41.State, parked: []dsv41.Parked, gens: []const u64, t1s: []const u32, out: []dsv41.DsparkDraft) !void {
    const n = gens.len;
    dsv41.clusterHooks().tp_draft = &Sent.send;
    defer dsv41.clusterHooks().tp_draft = null;
    Group.join(ranks);
    defer Group.leave();
    var have = false;
    defer if (have) freeDrafts(gpa, out[0..n]);
    // Three stages a sum each for attention and MoE, then a gather a Markov step.
    const collectives = 2 * ms.whole.m.n_mtp + ms.whole.m.ds_block;
    for (0..collectives + 2) |_| for (0..ranks) |r| {
        Group.begin(r);
        if (r == 0) {
            if (have) freeDrafts(gpa, out[0..n]);
            have = false;
            try dsv41.dsparkDraftBatched(ms.rank[0].m, gpa, states[0], gens, t1s, false, out[0..n]);
            have = true;
            try testing.expectEqualSlices(u64, gens, Sent.gens[0..Sent.n]);
            for (states[0], Sent.p0s[0..n]) |st, p| try testing.expectEqual(st.n, p);
            continue;
        }
        try dsv41.tpWorkerDraft(ms.rank[r].m, gpa, &parked[r], Sent.gens[0..n], Sent.p0s[0..n], Sent.t1s[0..n]);
    };
    try Group.expectSettled();
    have = false;
}

/// The same ids; the confidences within the four-rank trunk's bar, against the largest one.
fn expectSameIds(want: []const dsv41.DsparkDraft, got: []const dsv41.DsparkDraft) !void {
    const tol: f32 = if (device.naxAvailable()) 1e-2 else 1e-3;
    var peak: f32 = 0;
    for (want) |a| for (a.confidence) |u| {
        peak = @max(peak, @abs(u));
    };
    try testing.expect(peak > 0);
    for (want, got) |a, b| {
        try testing.expectEqualSlices(u32, a.ids, b.ids);
        try testing.expectEqual(a.len, b.len);
        for (a.confidence, b.confidence) |u, v| try testing.expect(@abs(u - v) <= tol * peak);
    }
}

fn expectSameWindows(gpa: std.mem.Allocator, a: *const dsv41.State, b: *const dsv41.State) !void {
    try testing.expectEqual(a.n, b.n);
    for (a.ds, b.ds) |x, y| {
        const u = try dsv41.hostF32(x.buf);
        defer gpa.free(u);
        const v = try dsv41.hostF32(y.buf);
        defer gpa.free(v);
        try testing.expectEqualSlices(f32, u, v);
    }
}

/// Opens a verify of `rows` positions on `st`, its main rows random, as a verify step leaves it.
fn openVerify(fx: *fixture.Fixture, st: *dsv41.State, rows: usize) !void {
    const c = &fx.m.cfg;
    try dsv41.armVerify(st, fx.m.n_layers);
    st.verify.?.mh = try fx.normal(&.{ @intCast(rows), @intCast(c.dsv4_n_dspark_target_layers * c.hidden_size) }, fx.m.act, 0, 1);
    st.n += rows;
}

test "dsv41 draft shard: four ranks' tails over the same stream draft the whole head's ids and confidences, bit for bit" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var ms = try Models.init(gpa, .{});
    defer ms.deinit();
    const B = ms.whole.m.ds_block;
    for ([_]usize{ 1, 3 }) |N| {
        Group.join(ranks);
        defer Group.leave();
        const x = try ms.whole.normal(&.{ @intCast(N * B), @intCast(ms.whole.m.cfg.dsv4_hc_mult), @intCast(ms.whole.m.cfg.hidden_size) }, .float32, 0, 1);
        defer _ = mlx.mlx_array_free(x);
        const pre = try ms.whole.cpre(N * B);
        defer _ = mlx.mlx_array_free(pre);
        const firsts = [_]i32{ 17, 900, 4095 };
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const f = try sc.ints(firsts[0..N], &.{@intCast(N)});
        const want = try dsv41.draftTail(&sc, ms.whole.m, x, pre, f, true, false);
        const want_ids = try dsv41.hostI32(want.ids);
        defer gpa.free(want_ids);
        const want_conf = try dsv41.hostF32(want.conf);
        defer gpa.free(want_conf);
        var got: dsv41.BatchGraph = undefined;
        for (0..B + 2) |_| for (0..ranks) |r| {
            Group.begin(r);
            const g = try dsv41.draftTail(&sc, ms.rank[r].m, x, pre, f, true, false);
            if (r == 0) got = g;
        };
        try Group.expectSettled();
        const got_ids = try dsv41.hostI32(got.ids);
        defer gpa.free(got_ids);
        const got_conf = try dsv41.hostF32(got.conf);
        defer gpa.free(got_conf);
        try testing.expectEqualSlices(i32, want_ids, got_ids);
        try testing.expectEqualSlices(f32, want_conf, got_conf);
    }
}

test "dsv41 draft shard: the gathered argmax is the whole row's, the first of equal values across and within ranks" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var ms = try Models.init(gpa, .{ .experts = 4, .vocab = 4093, .ctx = 8 });
    defer ms.deinit();
    Group.join(ranks);
    defer Group.leave();
    const V = ms.whole.m.vocab;
    // Row 0: the max at ids 1500 and 3500 (ranks 1 and 3); row 1: at ids 10 and 11 (rank 0); row 2: the last id alone.
    const whole = try gpa.alloc(f32, 3 * V);
    defer gpa.free(whole);
    for (whole, 0..) |*v, k| v.* = @floatFromInt((k * 7919) % 1000);
    for ([_][2]usize{ .{ 0, 1500 }, .{ 0, 3500 }, .{ 1, 10 }, .{ 1, 11 }, .{ 2, V - 1 } }) |at| whole[at[0] * V + at[1]] = 5000;
    const want = [_]i32{ 1500, 10, @intCast(V - 1) };
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const rows = try sc.keep(mlx.mlx_array_new_data(whole.ptr, &[_]c_int{ 3, @intCast(V) }, 2, .float32));
    const alone = try dsv41.hostI32(try dsv41.vocabArgmax(&sc, ms.whole.m, rows));
    defer gpa.free(alone);
    try testing.expectEqualSlices(i32, &want, alone);
    var got: A = undefined;
    for (0..3) |_| for (0..ranks) |r| {
        Group.begin(r);
        const p = ms.rank[r].m.head_part.?;
        got = try dsv41.vocabArgmax(&sc, ms.rank[r].m, try sc.slice(rows, 1, p.lo, p.hi));
    };
    try Group.expectSettled();
    const ids = try dsv41.hostI32(got);
    defer gpa.free(ids);
    try testing.expectEqualSlices(i32, &want, ids);
}

test "dsv41 draft shard: four ranks draft the unsharded draft's ids, one state and two, through the workers' entry, with the same collectives in the same order" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var ms = try Models.init(gpa, .{});
    defer ms.deinit();
    const fxs = ms.all();
    var seconds: [ranks + 1]dsv41.State = undefined;
    var made: usize = 0;
    defer for (seconds[0..made]) |*st| st.deinit();
    for (fxs, &seconds) |fx, *st| {
        fx.m.ds_want_conf = fx == ms.whole or fx == ms.rank[0];
        st.* = try dsv41.initState(fx.m, gpa);
        made += 1;
        try fx.fillWindows(st, 40);
    }
    // Each worker knows the second state by its gen, parked.
    const second_gen: u64 = 7;
    var parked: [ranks]dsv41.Parked = undefined;
    for (&parked) |*p| p.* = .init(gpa);
    defer for (&parked, 0..) |*p, r| if (r == 0) p.deinit() else dsv41.dropParked(p);
    for (1..ranks) |r| try parked[r].put(second_gen, seconds[r + 1]);
    made = 2;
    var worker_second: [ranks]*dsv41.State = undefined;
    worker_second[0] = &seconds[1];
    for (1..ranks) |r| worker_second[r] = parked[r].getPtr(second_gen).?;
    const first_gen = ms.rank[0].m.state_gen;

    for (0..3) |round| {
        // Round 1 drafts after a verify of 4 rows that kept 3: rank 0 finishes it, a worker follows it in the draft.
        if (round == 1) for (fxs, 0..) |fx, i| {
            const st = if (i == 0) &seconds[0] else worker_second[i - 1];
            try openVerify(fx, st, 4);
            if (i <= 1) try dsv41.finishVerify(fx.m, gpa, st, 3);
        };
        const t1s = [_]u32{ @intCast(7 + 31 * round), @intCast(1000 + 13 * round) };
        for ([_]usize{ 1, 2 }) |n| {
            var want: [2]dsv41.DsparkDraft = undefined;
            try dsv41.dsparkDraftBatched(ms.whole.m, gpa, (&[_]*dsv41.State{ ms.whole.state(), &seconds[0] })[0..n], (&[_]u64{ 0, 0 })[0..n], t1s[0..n], false, want[0..n]);
            defer freeDrafts(gpa, want[0..n]);
            var states: [ranks][]const *dsv41.State = undefined;
            var lists: [ranks][2]*dsv41.State = undefined;
            for (&lists, &states, 0..) |*l, *s, r| {
                l.* = .{ ms.rank[r].state(), worker_second[r] };
                s.* = l[0..n];
            }
            var got: [2]dsv41.DsparkDraft = undefined;
            try groupDraft(gpa, &ms, states, &parked, (&[_]u64{ first_gen, second_gen })[0..n], t1s[0..n], got[0..n]);
            defer freeDrafts(gpa, got[0..n]);
            try expectSameIds(want[0..n], got[0..n]);
        }
        try expectSameWindows(gpa, &seconds[0], worker_second[0]);
        for (1..ranks) |r| try expectSameWindows(gpa, worker_second[0], worker_second[r]);
        for (fxs) |fx| fx.state().n += 1;
        seconds[0].n += 1;
        for (worker_second) |st| st.n += 1;
    }
}

test "dsv41 draft shard: what a snapshot read from disk sends is the same on rank 0 and a worker, the DSpark windows included" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var ms = try Models.init(gpa, .{ .experts = 4, .ctx = 8 });
    defer ms.deinit();
    for ([_]usize{ 1, 130, 4096 }) |n| {
        const a = try dsv41.tpLoadPlan(ms.rank[0].m, n);
        for (ms.rank[1..]) |fx| {
            const b = try dsv41.tpLoadPlan(fx.m, n);
            try testing.expectEqual(a.len, b.len);
            var windows: usize = 0;
            for (a.specs(), b.specs()) |x, y| {
                try testing.expectEqualDeep(x, y);
                windows += @intFromBool(x.kind == .ds);
            }
            try testing.expectEqual(ms.rank[0].m.n_mtp, windows);
        }
    }
}

test "dsv41 draft shard: a sharded draft refuses to copy logits and a state other ranks do not know by gen" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.initDraft(gpa, .{ .experts = 4, .vocab = 4096, .ctx = 8, .act = .float32, .tp = .{ .rank = 0, .size = ranks } });
    defer fx.deinit();
    try testing.expect(fx.m.draft_shard);
    var out: [1]dsv41.DsparkDraft = undefined;
    try testing.expectError(error.Dsv41DraftShardLogits, dsv41.dsparkDraftBatched(fx.m, gpa, &.{fx.state()}, &.{0}, &.{1}, true, &out));
    var other = try dsv41.initState(fx.m, gpa);
    defer other.deinit();
    try testing.expectError(error.Dsv41TpStateNotLive, dsv41.dsparkDraft(fx.m, gpa, &other, 1));
}

fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// A rank alone: each sum returns its own part and each gather its own part once a rank, so the time is the rank's
/// own work at its shapes, without the link.
const Alone = struct {
    fn sum(x: A) anyerror!A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }
    fn gather(x: A) anyerror!A {
        const xs: [ranks]A = @splat(x);
        const v = mlx.mlx_vector_array_new_data(&xs, ranks);
        defer _ = mlx.mlx_vector_array_free(v);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&o, v, 0, mlx.gpuStream()));
        return o;
    }
    fn send(_: []const u64, _: []const usize, _: []const u32) anyerror!void {}
};

fn draftMs(fx: *fixture.Fixture, n: usize) !f64 {
    const gpa = fx.gpa;
    var second = try dsv41.initState(fx.m, gpa);
    defer second.deinit();
    try fx.fillWindows(&second, fx.state().n);
    const states = [_]*dsv41.State{ fx.state(), &second };
    const reps = 40;
    var t: [reps]f64 = undefined;
    for (0..reps + 5) |i| {
        var out: [2]dsv41.DsparkDraft = undefined;
        const t0 = nowUs();
        try dsv41.dsparkDraftBatched(fx.m, gpa, states[0..n], (&[_]u64{ 0, 1 })[0..n], (&[_]u32{ 42, 43 })[0..n], false, out[0..n]);
        if (i >= 5) t[i - 5] = @floatFromInt(nowUs() - t0);
        freeDrafts(gpa, out[0..n]);
    }
    return median(&t) / 1e3;
}

test "dsv41 draft ubench: one draft of the release's DSpark stages whole, and as one rank of four's share (DSV41_DEC_UBENCH)" {
    if (!decode.ubenchOn() or gpuUnavailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    dsv41.clusterHooks().tp_sum = &Alone.sum;
    dsv41.clusterHooks().tp_gather = &Alone.gather;
    dsv41.clusterHooks().tp_draft = &Alone.send;
    defer {
        dsv41.clusterHooks().tp_sum = null;
        dsv41.clusterHooks().tp_gather = null;
        dsv41.clusterHooks().tp_draft = null;
    }
    for ([_]?u32{ null, 0, 3 }) |rank| {
        const fx = try fixture.Fixture.initDraft(gpa, .{ .tp = if (rank) |r| .{ .rank = r, .size = ranks } else null });
        defer fx.deinit();
        const stages = fx.m.n_mtp;
        // Twice each, the first warming the kernels; no stages is the tail alone.
        for ([_]usize{ stages, stages, 0 }) |n_stages| for ([_]usize{ 1, 2 }) |n| {
            fx.m.n_mtp = n_stages;
            defer fx.m.n_mtp = stages;
            const ms = try draftMs(fx, n);
            if (rank) |r| {
                std.debug.print("[draft-ubench] tp4 rank={d} stages={d} states={d} ms_per_draft={d:.3} (own work, collectives stubbed)\n", .{ r, n_stages, n, ms });
            } else std.debug.print("[draft-ubench] whole stages={d} states={d} ms_per_draft={d:.3}\n", .{ n_stages, n, ms });
        };
    }
}
