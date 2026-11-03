//! A0: the decode's first DSpark cycle against its warm cycles, in profile builds only (`dsv41_decode_timers.enabled`;
//! in every other build each entry below compiles to nothing and the launch paths carry no call).
//!
//! What it counts:
//!   - per routed layer call: the routing barrier's wait, the route, the MoE's build and commit (hit, hoist and miss
//!     waves), and the distinct experts the call missed; the first cycle apart, the warm cycles summed;
//!   - per layer, the demand reads' drive time: the stream's read gauge from one barrier's end to the next layer's
//!     (the reads a route issues land before the next barrier returns), against that barrier;
//!   - the prompt's last `tail_rows` rows' routed experts per layer, and in the first cycle how many of each layer's
//!     misses they hold and how many reads they would cost (A0's first-verify reads at the grow);
//!   - the decode timers' buckets at the first cycle's end;
//!   - A0 (c)'s D1: the first cycle's draft block run twice (`Loop.droppedDraft`), the first dropped, each block's build,
//!     wait and the bytes MLX took for it, after the state the draft's eval would realise (evaluated apart);
//!   - the first dispatches of each phase (construction, the prompt pass, the first cycle, the warm cycles): a
//!     registry kernel's variant (MLX builds one library per kernel, template and input binding: scalar, `constant`
//!     under 8 elements, else `device`) and a compiled region's input signature (MLX traces one per shapes and dtypes).
//! Printed once after the decode (`DECODE_FIRST`).
const std = @import("std");
const mlx = @import("sdk").mlx;
const dt = @import("decode_timers.zig");

pub const enabled = dt.enabled;

/// The process's phase: construction (its self-checks and warm-up), the prompt pass (through the phase change), the
/// decode's first cycle, its later cycles.
pub const Phase = enum(u8) { build, prompt, cycle1, warm };
pub var phase: Phase = .build;

pub const max_layers = @import("decode_timers.zig").max_layers;
pub const max_experts = 512;
pub const Layer = struct { calls: u64 = 0, barrier_ns: u64 = 0, route_ns: u64 = 0, moe_ns: u64 = 0, misses: u64 = 0, read_ns: u64 = 0 };
/// Per routed layer: [0] the first cycle, [1] the warm cycles summed.
pub var layers: [2][max_layers]Layer = @splat(@splat(.{}));
/// The decode timers' buckets and routed calls at the first cycle's end.
pub var cycle1_ns: @TypeOf(dt.ns) = @splat(0);
pub var cycle1_calls: u64 = 0;
/// The previous routed call's layer and the read gauge at its barrier's end (`call`).
var last_layer: ?u32 = null;
var last_wall: u64 = 0;

/// The prompt rows whose experts make the tail set: the prompt's last rows per layer (its last call's).
pub const tail_rows = 8;
pub const Tail = struct { size: u64 = 0, reads: u64 = 0, hits: u64 = 0 };
var tail: [max_layers]std.StaticBitSet(max_experts) = @splat(.empty);
/// Per layer in the first cycle: the tail set's size, its experts not resident (the reads it costs), and the
/// call's misses it holds.
pub var tails: [max_layers]Tail = @splat(.{});

/// A0 (c)'s D1: one of the first cycle's draft blocks. Its graph's build and its eval's wait (ns), MLX's cache bytes
/// before it, and the bytes MLX took from the system for it (`fresh_bytes`: active + cache after its eval, less before).
pub const DraftBlock = struct { build_ns: u64 = 0, wait_ns: u64 = 0, cache_before: u64 = 0, fresh_bytes: u64 = 0 };
/// [0] the dropped block, [1] the cycle's own.
pub var cycle1_draft: [2]DraftBlock = @splat(.{});
/// The main row and the stage windows the first draft's eval would realise (the prompt's seed), evaluated first.
pub var cycle1_pending_ns: u64 = 0;
var drafts_seen: u8 = 0;

/// First dispatches counted per phase; the first cycle's named.
pub var new_per_phase: [4]u32 = @splat(0);
pub const Named = struct { what: []const u8, key: u64 };
pub var cycle1_new: [96]Named = undefined;
pub var n_cycle1_new: usize = 0;
var seen: [16384]u64 = @splat(0);

/// A launch config's template key (`exl3_kernels.launchKey`, kept by `Bound.prepare`).
pub const TKey = if (enabled) u64 else void;

/// The prompt pass begins (construction's dispatches are behind it).
pub fn startPrompt() void {
    if (comptime !enabled) return;
    phase = .prompt;
}

/// The decode begins: the per-layer counts cleared, the first cycle's dispatches counted apart.
pub fn startDecode() void {
    if (comptime !enabled) return;
    layers = @splat(@splat(.{}));
    cycle1_ns = @splat(0);
    cycle1_calls = 0;
    n_cycle1_new = 0;
    last_layer = null;
    tails = @splat(.{});
    cycle1_draft = @splat(.{});
    cycle1_pending_ns = 0;
    drafts_seen = 0;
    phase = .cycle1;
}

/// The next draft block is the decode's first: D1 runs it twice (once dropped).
pub fn firstDraft() bool {
    if (comptime !enabled) return false;
    return phase == .cycle1 and drafts_seen == 0;
}

/// MLX's active and cached bytes now on the device backend; zeros on the trace backend (no device, no MLX call).
pub fn memNow(comptime on_device: bool) [2]u64 {
    if (comptime !enabled or !on_device) return .{ 0, 0 };
    var act: usize = 0;
    var cached: usize = 0;
    _ = mlx.mlx_get_active_memory(&act);
    _ = mlx.mlx_get_cache_memory(&cached);
    return .{ act, cached };
}

/// One of the first cycle's draft blocks (0: the dropped one, 1: the cycle's own): its build and wait, and MLX's
/// (active, cache) bytes before it and after its eval.
pub fn recordDraft(i: u1, build_ns: u64, wait_ns: u64, before: [2]u64, after: [2]u64) void {
    if (comptime !enabled) return;
    cycle1_draft[i] = .{ .build_ns = build_ns, .wait_ns = wait_ns, .cache_before = before[1], .fresh_bytes = (after[0] + after[1]) -| (before[0] + before[1]) };
    drafts_seen = @max(drafts_seen, @as(u8, i) + 1);
}

/// A prompt call's routed ids (`n` rows x `k`, row order): its last `tail_rows` rows' experts become the layer's
/// tail set (a later call of the layer replaces it, so the prompt's last call's rows stand).
pub fn tailRecord(layer: u32, ids: []const u16, n: u32, k: u32) void {
    if (comptime !enabled) return;
    if (phase != .prompt or layer >= max_layers or n == 0) return;
    const r = @min(n, tail_rows);
    tail[layer] = .empty;
    for (ids[(n - r) * k .. n * k]) |e| {
        if (e < max_experts) tail[layer].set(e);
    }
}

/// The tail set's experts at `layer` not resident there now (before its first-cycle route): the reads it costs.
pub fn tailReads(layer: u32, ctx: anytype, comptime resident: fn (@TypeOf(ctx), u16) bool) u64 {
    if (comptime !enabled) return 0;
    if (phase != .cycle1 or layer >= max_layers) return 0;
    var n: u64 = 0;
    var it = tail[layer].iterator(.{});
    while (it.next()) |e| n += @intFromBool(!resident(ctx, @intCast(e)));
    return n;
}

/// A DSpark round has ended: after the first, its buckets are kept and the warm cycles begin.
pub fn endCycle() void {
    if (comptime !enabled) return;
    if (phase != .cycle1) return;
    cycle1_ns = dt.ns;
    cycle1_calls = dt.routed_calls;
    phase = .warm;
}

/// One routed layer call at decode width (`Experts.run`'s stamps, ns): its ids and their waves (its misses), the
/// stream's read gauge at its barrier's end (`read_wall_ns`; the previous layer's demand reads landed by then), and
/// in the first cycle the tail set's reads (`tailReads`, before the route).
pub fn call(layer: u32, barrier_ns: u64, route_ns: u64, moe_ns: u64, ids: []const u16, waves: []const u8, read_wall_ns: u64, tail_reads: u64) void {
    if (comptime !enabled) return;
    const w: usize = switch (phase) {
        .cycle1 => 0,
        .warm => 1,
        else => return,
    };
    if (layer >= max_layers) return;
    // the gauge between the previous layer's barrier and this one's: the previous route's demand reads
    if (last_layer) |ll| if (layer == ll + 1) {
        layers[w][ll].read_ns += read_wall_ns -| last_wall;
    };
    last_layer = layer;
    last_wall = read_wall_ns;
    var m: std.StaticBitSet(max_experts) = .empty;
    for (ids, waves) |e, wv| {
        if (wv > 0 and e < max_experts) m.set(e);
    }
    const l = &layers[w][layer];
    l.calls += 1;
    l.barrier_ns += barrier_ns;
    l.route_ns += route_ns;
    l.moe_ns += moe_ns;
    l.misses += m.count();
    if (w == 0) tails[layer] = .{ .size = tail[layer].count(), .reads = tail_reads, .hits = m.intersectWith(tail[layer]).count() };
}


/// A registry kernel dispatched (`Bound.applyPrepared`, through the observer the kernel set installs from the
/// backend's profile hook): its variant counted in the phase that first dispatches it.
pub fn kernel(name: []const u8, tkey: TKey, inputs: []const mlx.mlx_array) void {
    if (comptime !enabled) return;
    var h = std.hash.Wyhash.init(tkey);
    h.update(name);
    for (inputs) |x| {
        const d: c_int = @backingInt(mlx.mlx_array_dtype(x));
        const bind: u8 = if (mlx.mlx_array_ndim(x) == 0) 's' else if (mlx.mlx_array_size(x) < 8) 'c' else 'd';
        h.update(std.mem.asBytes(&d));
        h.update(&.{bind});
    }
    note(name, h.final());
}

/// A compiled region applied (`MlxOps.tape`): its input signature (shapes and dtypes) counted in the phase that first
/// traces it.
pub fn region(name: []const u8, ctx: usize, inputs: []const mlx.mlx_array) void {
    if (comptime !enabled) return;
    var h = std.hash.Wyhash.init(ctx);
    h.update(name);
    for (inputs) |x| {
        const d: c_int = @backingInt(mlx.mlx_array_dtype(x));
        h.update(std.mem.asBytes(&d));
        const nd = mlx.mlx_array_ndim(x);
        if (nd > 0) h.update(std.mem.sliceAsBytes(mlx.mlx_array_shape(x)[0..nd]));
        h.update("|");
    }
    note(name, h.final());
}

fn note(name: []const u8, hash: u64) void {
    const key = hash | 1;
    var i: usize = @intCast(key % seen.len);
    for (0..seen.len) |_| {
        if (seen[i] == key) return;
        if (seen[i] == 0) break;
        i = (i + 1) % seen.len;
    } else return;
    seen[i] = key;
    new_per_phase[@backingInt(phase)] += 1;
    if (phase == .cycle1 and n_cycle1_new < cycle1_new.len) {
        cycle1_new[n_cycle1_new] = .{ .what = name, .key = key };
        n_cycle1_new += 1;
    }
}

/// `DECODE_FIRST {...}`: the first dispatches per phase and the first cycle's new ones by name; the timers' buckets
/// for the first cycle and the warm cycles' mean; per routed layer the first cycle against the warm cycles' mean; the
/// first cycle's routed misses beside its stream count (`stream_misses`: every read of the cycle, when known).
pub fn line(buf: []u8, n_layers: u32, stream_misses: ?u64) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    write(&w, @min(n_layers, max_layers), stream_misses) catch return buf[0..0];
    return w.buffered();
}

fn write(w: *std.Io.Writer, n: u32, stream_misses: ?u64) !void {
    const warm: u64 = dt.cycles -| 1;
    const ms = struct {
        fn of(x: u64, per: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6 / @as(f64, @floatFromInt(@max(per, 1)));
        }
    }.of;
    try w.print("DECODE_FIRST {{\"cycles\": {d}, \"new_dispatches\": {{\"build\": {d}, \"prompt\": {d}, \"cycle1\": {d}, \"warm\": {d}}}, \"cycle1_new\": [", .{ dt.cycles, new_per_phase[0], new_per_phase[1], new_per_phase[2], new_per_phase[3] });
    for (cycle1_new[0..n_cycle1_new], 0..) |x, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ", ", x.what });
    try w.writeAll("], \"cycle1_ms\": {");
    inline for (@typeInfo(dt.Bucket).@"enum".field_names, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, ms(cycle1_ns[i], 1) });
    try w.writeAll("}, \"warm_ms\": {");
    inline for (@typeInfo(dt.Bucket).@"enum".field_names, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, ms(dt.ns[i] -| cycle1_ns[i], warm) });
    var routed: [2]u64 = @splat(0);
    for (0..2) |c| {
        for (layers[c][0..n]) |l| routed[c] += l.misses;
    }
    try w.print("}}, \"cycle1_routed_calls\": {d}, \"cycle1_routed_misses\": {d}, \"warm_routed_misses_per_cycle\": {d:.2}", .{ cycle1_calls, routed[0], @as(f64, @floatFromInt(routed[1])) / @as(f64, @floatFromInt(@max(warm, 1))) });
    if (stream_misses) |s| try w.print(", \"cycle1_stream_misses\": {d}", .{s});
    const Field = enum { barrier, route, moe, misses, read_busy };
    inline for (.{ Field.barrier, Field.route, Field.moe, Field.misses, Field.read_busy }) |f| {
        inline for (.{ "cycle1", "warm" }, 0..) |label, c| {
            try w.print(", \"{s}_{s}{s}\": [", .{ label, @tagName(f), if (f == .misses) "" else "_ms" });
            for (layers[c][0..n], 0..) |l, i| {
                const per: u64 = if (c == 0) 1 else warm;
                const v: f64 = switch (f) {
                    .barrier => ms(l.barrier_ns, per),
                    .route => ms(l.route_ns, per),
                    .moe => ms(l.moe_ns, per),
                    .misses => @as(f64, @floatFromInt(l.misses)) / @as(f64, @floatFromInt(@max(per, 1))),
                    .read_busy => ms(l.read_ns, per),
                };
                try w.print("{s}{d:.2}", .{ if (i == 0) "" else ", ", v });
            }
            try w.writeAll("]");
        }
    }
    // the tail set (the prompt's last rows' experts) against the first cycle's misses
    var tt: Tail = .{};
    for (tails[0..n]) |t| {
        tt.size += t.size;
        tt.reads += t.reads;
        tt.hits += t.hits;
    }
    try w.print(", \"tail_rows\": {d}, \"cycle1_tail_size\": {d}, \"cycle1_tail_reads\": {d}, \"cycle1_tail_hits\": {d}, \"cycle1_tail_recall\": {d:.3}", .{ tail_rows, tt.size, tt.reads, tt.hits, @as(f64, @floatFromInt(tt.hits)) / @as(f64, @floatFromInt(@max(routed[0], 1))) });
    inline for (.{ "size", "reads", "hits" }) |name| {
        try w.print(", \"cycle1_tail_{s}_by_layer\": [", .{name});
        for (tails[0..n], 0..) |t, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", @field(t, name) });
        try w.writeAll("]");
    }
    // D1: the pending state, then the dropped block and the cycle's own (zeros when the first cycle drafted once)
    try w.print(", \"cycle1_pending_ms\": {d:.3}", .{ms(cycle1_pending_ns, 1)});
    inline for (.{ "first", "second" }, 0..) |label, i| {
        const b = cycle1_draft[i];
        try w.print(", \"cycle1_draft_{s}\": {{\"build_ms\": {d:.3}, \"wait_ms\": {d:.3}, \"cache_before\": {d}, \"fresh_bytes\": {d}}}", .{ label, ms(b.build_ns, 1), ms(b.wait_ns, 1), b.cache_before, b.fresh_bytes });
    }
    try w.writeAll("}");
}

test "dsv41 decode first: a variant or signature counts once, in the phase that first dispatches it; the first cycle's are named" {
    if (comptime !enabled) return error.SkipZigTest;
    phase = .build;
    new_per_phase = @splat(0);
    seen = @splat(0);
    note("a", 0x10);
    note("a", 0x10);
    phase = .prompt;
    note("b", 0x20);
    startDecode();
    note("a", 0x10);
    note("c", 0x30);
    note("d", 0x40);
    endCycle();
    note("e", 0x50);
    try std.testing.expectEqual([4]u32{ 1, 1, 2, 1 }, new_per_phase);
    try std.testing.expectEqual(@as(usize, 2), n_cycle1_new);
    try std.testing.expectEqualStrings("c", cycle1_new[0].what);
    // past the first cycle a call counts with the warm ones
    call(2, 5_000_000, 1_000_000, 2_000_000, &.{ 3, 3, 7, 9 }, &.{ 1, 1, 0, 2 }, 0, 0);
    try std.testing.expectEqual(@as(u64, 0), layers[0][2].calls);
    try std.testing.expectEqual(@as(u64, 2), layers[1][2].misses);
    // the prompt's last 8 rows (of 10, one expert each) are layer 2's tail set
    phase = .prompt;
    tailRecord(2, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 10, 1);
    startDecode();
    // layers 1 and 2 in one verify: the gauge between their barriers is layer 1's reads; layer 2 misses 3 and 9,
    // both in its tail set, whose 5 non-resident experts are its reads
    call(1, 1_000_000, 0, 0, &.{4}, &.{0}, 1_000_000, 0);
    call(2, 5_000_000, 1_000_000, 2_000_000, &.{ 3, 3, 7, 9 }, &.{ 1, 1, 0, 2 }, 4_000_000, 5);
    try std.testing.expectEqual(@as(u64, 3_000_000), layers[0][1].read_ns);
    try std.testing.expectEqual(@as(u64, 2), layers[0][2].misses);
    try std.testing.expectEqual(Tail{ .size = 8, .reads = 5, .hits = 2 }, tails[2]);
    var buf: [16384]u8 = undefined;
    const l = line(&buf, 4, 300);
    try std.testing.expect(std.mem.startsWith(u8, l, "DECODE_FIRST {\"cycles\": "));
    try std.testing.expect(std.mem.indexOf(u8, l, "\"cycle1_barrier_ms\": [0.00, 1.00, 5.00, 0.00]") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "\"cycle1_read_busy_ms\": [0.00, 3.00, 0.00, 0.00]") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "\"cycle1_stream_misses\": 300") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "\"cycle1_tail_size\": 8, \"cycle1_tail_reads\": 5, \"cycle1_tail_hits\": 2, \"cycle1_tail_recall\": 1.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "\"cycle1_tail_hits_by_layer\": [0, 0, 2, 0]") != null);
    // D1: the first cycle's first draft is doubled once; the bytes MLX took are active + cache after less before
    try std.testing.expect(firstDraft());
    recordDraft(0, 3_000_000, 34_000_000, .{ 100, 0 }, .{ 100, 17 });
    try std.testing.expect(!firstDraft());
    recordDraft(1, 1_000_000, 7_000_000, .{ 100, 17 }, .{ 103, 14 });
    cycle1_pending_ns = 2_000_000;
    const l2 = line(&buf, 4, 300);
    try std.testing.expect(std.mem.indexOf(u8, l2, "\"cycle1_pending_ms\": 2.000, \"cycle1_draft_first\": {\"build_ms\": 3.000, \"wait_ms\": 34.000, \"cache_before\": 0, \"fresh_bytes\": 17}, \"cycle1_draft_second\": {\"build_ms\": 1.000, \"wait_ms\": 7.000, \"cache_before\": 17, \"fresh_bytes\": 0}}") != null);
    endCycle();
    try std.testing.expect(!firstDraft());
    phase = .build;
}
