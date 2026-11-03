//! sdk_ext.expert.lookahead's tests (the Python lane's oracle traces), in the unit tests; the names it reads are the SDK's.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const lookahead = @import("../sdk_ext.zig").expert.lookahead;
const expert_policy = @import("../sdk_ext.zig").expert.policy;
const LayerPolicy = expert_policy.LayerPolicy;
pub const routed_top_k = @import("bank.zig").routed_top_k;
pub const min_k = routed_top_k;
pub const max_k = lookahead.max_k;
pub const max_budget = lookahead.max_budget;
pub const max_rows = lookahead.max_rows;
pub const max_candidates = lookahead.max_candidates;
pub const Selector = lookahead.SelectorOf(routed_top_k);

const testing = std.testing;

/// A policy holding exactly `residents` (prefill-admitted, capacity = their count).
fn policyWith(a: std.mem.Allocator, n_experts: u32, residents: []const u16) !LayerPolicy {
    var p = try LayerPolicy.init(a, n_experts, @intCast(residents.len));
    errdefer p.deinit(a);
    if (residents.len > 0) {
        var plan: expert_policy.Plan = .{};
        p.plan(residents, .prefill, &plan);
    }
    return p;
}

fn rowsOf(comptime n_experts: usize, comptime rows: usize, spec: [rows][]const struct { u16, f32 }) [rows * n_experts]f32 {
    var s: [rows * n_experts]f32 = @splat(0);
    for (spec, 0..) |row, r| for (row) |ev| {
        s[r * n_experts + ev[0]] = ev[1];
    };
    return s;
}

test "dsv41 lookahead: the union is ordered by best rank over rows, then score" {
    const a = testing.allocator;
    var sel = try Selector.init(a, 16, 8, std.math.inf(f32), 4);
    defer sel.deinit(a);
    var none = try policyWith(a, 16, &.{});
    defer none.deinit(a);
    const s = rowsOf(16, 2, .{
        &.{ .{ 3, 0.9 }, .{ 7, 0.8 }, .{ 1, 0.7 }, .{ 9, 0.6 }, .{ 4, 0.5 }, .{ 2, 0.4 }, .{ 11, 0.3 }, .{ 5, 0.2 }, .{ 13, 0.1 } },
        &.{ .{ 7, 0.95 }, .{ 5, 0.85 }, .{ 3, 0.75 }, .{ 12, 0.65 }, .{ 1, 0.55 }, .{ 0, 0.45 }, .{ 2, 0.35 }, .{ 8, 0.25 }, .{ 14, 0.15 } },
    });
    // rank 0: 7 (.95), 3 (.9); rank 1: 5 (.85), 7; rank 2: 3, 1; rank 3: 12, 9; rank 4: 1, 4; rank 5: 0, 2;
    // rank 6: 2, 11; rank 7: 5, 8. Expert 13 and 14 sit at rank 8: outside K = 8.
    var out: [max_candidates]u16 = undefined;
    try testing.expectEqualSlices(u16, &.{ 7, 3, 5, 1, 12, 9, 4, 0, 2, 11, 8 }, sel.select(&s, &none, &out));
    try testing.expectEqualSlices(u16, &.{ 7, 3, 5, 1 }, sel.select(&s, &none, out[0..sel.budget]));
    // Residents are skipped, not counted against the budget.
    var res = try policyWith(a, 16, &.{ 3, 5, 12 });
    defer res.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 7, 1, 9, 4 }, sel.select(&s, &res, out[0..sel.budget]));
}

test "dsv41 lookahead: the chosen experts' top-k positions come back with them" {
    const a = testing.allocator;
    var sel = try Selector.init(a, 16, 8, std.math.inf(f32), 4);
    defer sel.deinit(a);
    var res = try policyWith(a, 16, &.{ 3, 5, 12 });
    defer res.deinit(a);
    const s = rowsOf(16, 2, .{
        &.{ .{ 3, 0.9 }, .{ 7, 0.8 }, .{ 1, 0.7 }, .{ 9, 0.6 }, .{ 4, 0.5 }, .{ 2, 0.4 }, .{ 11, 0.3 }, .{ 5, 0.2 }, .{ 13, 0.1 } },
        &.{ .{ 7, 0.95 }, .{ 5, 0.85 }, .{ 3, 0.75 }, .{ 12, 0.65 }, .{ 1, 0.55 }, .{ 0, 0.45 }, .{ 2, 0.35 }, .{ 8, 0.25 }, .{ 14, 0.15 } },
    });
    const resident = lookahead.residentIn;
    var out: [max_budget]u16 = undefined;
    var ranks: [max_budget]u8 = undefined;
    try testing.expectEqualSlices(u16, &.{ 7, 1, 9, 4 }, sel.selectWith(&s, &res, resident, &out, &ranks));
    try testing.expectEqualSlices(u8, &.{ 0, 2, 3, 4 }, &ranks);
}

test "dsv41 lookahead: tau keeps a candidate within tau of its row's sixth score" {
    const a = testing.allocator;
    var none = try policyWith(a, 16, &.{});
    defer none.deinit(a);
    const s = rowsOf(16, 1, .{&.{ .{ 0, 1.0 }, .{ 1, 0.9 }, .{ 2, 0.8 }, .{ 3, 0.7 }, .{ 4, 0.6 }, .{ 5, 0.5 }, .{ 6, 0.45 }, .{ 7, 0.39 }, .{ 8, 0.38 } }});
    var out: [max_candidates]u16 = undefined;
    var t = try Selector.init(a, 16, 8, 0.1, 4);
    defer t.deinit(a);
    // Sixth score 0.5, threshold 0.4 in f32: 0.45 is kept, 0.39 is not; 0.38 (rank 8) is outside K.
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4, 5, 6 }, t.select(&s, &none, &out));
    var z = try Selector.init(a, 16, 8, 0.0, 4);
    defer z.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4, 5 }, z.select(&s, &none, &out));
    var neg = try Selector.init(a, 16, 8, -0.15, 4);
    defer neg.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3 }, neg.select(&s, &none, &out));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 5, 0, 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 13, 0, 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 8, std.math.nan(f32), 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 8, -std.math.inf(f32), 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 8, 0, 5));
}

test "dsv41 lookahead: a score tie breaks by expert id" {
    const a = testing.allocator;
    var none = try policyWith(a, 16, &.{});
    defer none.deinit(a);
    var sel = try Selector.init(a, 16, 6, std.math.inf(f32), 4);
    defer sel.deinit(a);
    const s = rowsOf(16, 1, .{&.{ .{ 9, 2.0 }, .{ 15, 2.0 }, .{ 4, 2.0 }, .{ 1, 1.0 }, .{ 10, 0.5 }, .{ 3, 0.5 }, .{ 7, 0.5 } }});
    var out: [max_candidates]u16 = undefined;
    // Three tied at the top, three tied for the last two places: the lower ids win both.
    try testing.expectEqualSlices(u16, &.{ 4, 9, 15, 1, 3, 7 }, sel.select(&s, &none, &out));
}

test "dsv41 lookahead: certain misses are the route's unique non-residents in route order" {
    const a = testing.allocator;
    var sel = try Selector.init(a, 16, 8, std.math.inf(f32), 2);
    defer sel.deinit(a);
    var res = try policyWith(a, 16, &.{ 2, 4 });
    defer res.deinit(a);
    var out: [max_candidates]u16 = undefined;
    const ids = [_]u16{ 5, 2, 9, 5, 4, 1, 9, 12 };
    try testing.expectEqualSlices(u16, &.{ 5, 9, 1, 12 }, sel.certainMisses(&ids, &res, &out));
    try testing.expectEqualSlices(u16, &.{ 5, 9 }, sel.certainMisses(&ids, &res, out[0..2]));
    // The plan's misses are the same experts in the same order (decode).
    var plan: expert_policy.Plan = .{};
    try res.grow(3);
    res.plan(&ids, .decode, &plan);
    try testing.expectEqualSlices(u16, &.{ 5, 9, 1, 12 }, plan.missesOf());
}

// DSV41_PHASE2_FIXTURE=<json from the reference runtime's dump_phase2_lookahead_fixture.py> (its scores file beside it)
test "dsv41 lookahead: the recorded trace selects exactly like the Python lane" {
    const path = std.mem.span(env.get(.DSV41_PHASE2_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const Call = struct { ids: []const u16, pre: []const u16, sel: []const []const u16, cand: []const []const u16 };
    const Config = struct { k: u32, tau: ?f32, budget: u32 };
    const TieCall = struct { call: u32, config: u32, sel: []const u16, cand: []const u16 };
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        rows: []const u32,
        configs: []const Config,
        scores_file: []const u8,
        scores_rows: u64,
        tie_calls: []const TieCall,
        calls: []const Call,
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var dir_buf: [1024]u8 = undefined;
    const scores_path = try std.fmt.bufPrint(&dir_buf, "{s}/{s}", .{ std.fs.path.dirname(path) orelse ".", f.scores_file });
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, scores_path, a, .limited(64 << 20));
    defer a.free(raw);
    try testing.expectEqual(f.scores_rows * f.experts * 4, raw.len);
    const scores = try a.alloc(f32, raw.len / 4);
    defer a.free(scores);
    for (scores, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));

    const policies = try a.alloc(LayerPolicy, f.layers);
    defer a.free(policies);
    var n_init: usize = 0;
    defer for (policies[0..n_init]) |*p| p.deinit(a);
    var plan: expert_policy.Plan = .{};
    for (0..f.layers) |l| {
        policies[l] = try LayerPolicy.init(a, f.experts, f.prefill_capacity[l]);
        n_init += 1;
        const p = &policies[l];
        p.prepareSeed(f.resident0[l]);
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        var i: usize = 0;
        while (i < sorted.len) : (i += f.transient) p.plan(sorted[i..@min(i + f.transient, sorted.len)], .prefill, &plan);
    }
    for (policies, f.decode_capacity) |*p, cap| try p.grow(cap);

    var sels: [3]Selector = undefined;
    try testing.expectEqual(@as(usize, 3), f.configs.len);
    var n_sel: usize = 0;
    defer for (sels[0..n_sel]) |*s| s.deinit(a);
    for (f.configs, &sels) |cf, *s| {
        s.* = try Selector.init(a, f.experts, cf.k, cf.tau orelse std.math.inf(f32), cf.budget);
        n_sel += 1;
    }
    var out: [max_candidates]u16 = undefined;
    var at: usize = 0;
    var n_pre: usize = 0;
    var n_equal: usize = 0;
    var n_tie: usize = 0;
    var tie_i: usize = 0;
    for (f.calls, 0..) |call, ci| {
        const cycle = ci / f.layers;
        const l = ci % f.layers;
        const pre = sels[0].certainMisses(call.ids, &policies[l], &out);
        try testing.expectEqualSlices(u16, call.pre, pre);
        n_pre += pre.len;
        if (l + 1 < f.layers) {
            const m = f.rows[cycle];
            const s = scores[at * f.experts ..][0 .. m * f.experts];
            at += m;
            for (&sels, 0..) |*sel, k| {
                var want_sel = call.sel[k];
                var want_cand = call.cand[k];
                if (tie_i < f.tie_calls.len and f.tie_calls[tie_i].call == ci and f.tie_calls[tie_i].config == k) {
                    want_sel = f.tie_calls[tie_i].sel;
                    want_cand = f.tie_calls[tie_i].cand;
                    tie_i += 1;
                    n_tie += 1;
                } else n_equal += 1;
                testing.expectEqualSlices(u16, want_sel, sel.select(s, &policies[l + 1], out[0..sel.budget])) catch |e| {
                    std.debug.print("call {d} (cycle {d}, layer {d}) config {d}: selection differs\n", .{ ci, cycle, l, k });
                    return e;
                };
                try testing.expectEqualSlices(u16, want_cand, sel.select(s, &policies[l + 1], &out));
            }
        } else for (call.sel) |w| try testing.expectEqual(@as(usize, 0), w.len);
        policies[l].plan(call.ids, .decode, &plan);
    }
    try testing.expectEqual(f.tie_calls.len, tie_i);
    try testing.expectEqual(f.scores_rows, at);
    std.debug.print("lookahead parity: {d} layer calls, {d} certain misses equal the lane's; {d} x 2 selections equal the lane's, {d} x 2 equal its tie-broken-by-id variant (exact score ties)\n", .{ f.calls.len, n_pre, n_equal, n_tie });
}
