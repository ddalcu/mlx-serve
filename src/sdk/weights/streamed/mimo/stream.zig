//! MiMo's six-component MXFP4 bank on the shared SDK stream. The SDK owns
//! residency, asynchronous reads, windows, pinning and release; this bridge
//! maps absolute model layers and exposes GatherQmm's {w, scales} bindings.
const std = @import("std");
const env = @import("mlx_host").env;
const sdk_ext = @import("../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const bank_mod = @import("bank.zig");
const expert_policy = sdk_ext.expert.policy;
const profile = @import("profile.zig");
const route_trace = @import("route_trace.zig");
const prefill_timeline = @import("prefill_timeline.zig");
const gpu_warm = @import("gpu_warm.zig");
const diag = @import("diag.zig");
/// MiMo's bridge over the shared SDK stream (`stream_source.For`): its bank module, GatherQmm's {w, scales} slot bindings
/// and its read pool's staging.
const Mimo = @import("stream_source.zig").For(bank_mod, .{ .Arrays = sdk_ext.quant.GatherQmm.Arrays, .staging_bytes = io_staging_bytes });
const S = Mimo.S;
pub const Stream = Mimo.Stream;
pub const Options = Mimo.Options;
pub const SlotMemory = Mimo.SlotMemory;
pub const LayerSlotBank = Mimo.LayerSlotBank;
pub const Route = Mimo.Route;
pub const Part = Mimo.Part;
pub const BankKind = Mimo.BankKind;
pub const SlotRef = Mimo.SlotRef;
pub const Error = Mimo.Error;
pub const Stats = Mimo.Stats;
pub const Gates = Mimo.Gates;
pub const max_route_ids = Mimo.max_route_ids;
pub const uses_reader = Mimo.uses_reader;
pub const BankArrays = bank_mod.BankArrays;
pub const BankArraysOf = bank_mod.BankArraysOf;

pub const io_workers: u32 = 4;
pub const io_staging_bytes: u64 = 9 << 20;
/// P1's read-ahead admits min(rows, (tickets - 2 * max_items) / 2) records (`Stream.readAheadSeed`): 512 covers
/// every MiMo row a layer can hold (248 >= 124 at the fill; 256 tickets admitted only 120).
pub const io_tickets: u32 = 512;
pub const records_per_part: u32 = 2;
pub const wide_depth: u8 = 2;
pub const transient_rows: u32 = wide_depth * max_route_ids;
pub const spec_threads: u32 = 2;
pub const spec_slots: u32 = 4;
/// Decode's transient rows under the phase-rows route (`Options.transient_release`): the boundary frees the prompt's
/// `transient_rows` and the grow allocates window 0 (`decodeRouteIds`) plus the stream's decode staging rows
/// (`Stream.grow`).
pub fn decodeTransientRows(window: DecodeWindow) u32 {
    return decodeRouteIds(window) + S.decode_staging_rows;
}
/// R3 (mimo/decode-rows-audit): decode's window 0, chosen once at construction (settings `"decode_window"`). `verify`
/// (the default) sizes it to the decode lane's widest route, `decode_lane_route_ids` = 4 rows x top-8 = 32 (a decode
/// plan's transient loads are its distinct non-admitted misses, at most its ids: `LayerPolicy.planWith`); `route` is
/// the control, the SDK's `max_route_ids` (48). Every decode-phase routed call is a decode-lane call (Module routes any
/// wider forward through the prompt phase first; `Source.forward` refuses a decode-phase wide call under `verify`).
pub const DecodeWindow = enum { verify, route, invalid };
/// The decode lane's widest route: `source.decode_lane_rows` (4: the MTP verify) x `source.top_k` (8), pinned there.
pub const decode_lane_route_ids: u32 = 4 * 8;
comptime {
    if (decode_lane_route_ids > max_route_ids) @compileError("mimo_v2: the decode lane's route would not fit one SDK route");
}
pub fn decodeRouteIds(window: DecodeWindow) u32 {
    return switch (window) {
        .verify => decode_lane_route_ids,
        .route, .invalid => max_route_ids,
    };
}
pub const lookahead = Mimo.lookahead;

pub const options = Mimo.options;

/// P7 (`settings.prefill_read_schedule`): the prompt pass's read schedule, fixed at construction. `current` (the control):
/// the stream's stock schedule. `early`: the stream's `overlap` schedule (`S.PromptReads`): every group of a prompt call
/// planned at its first route exactly as the stock schedule plans it, each read posted as soon as its row is free, the
/// read-ahead posted in rank order (head now, tail resolved by the plans: read once, the double reads dropped) and landed
/// only where a group needs it. Same records in the same rows before any kernel reads them: exact by construction.
/// `early_whole`: `early` with the WHOLE read-ahead posted with the attention, in rank order (`S.AheadPost.whole`):
/// p7-20261009 try 2 measured `early` moving ~0.84 s/pass of GPU idle into the router stage, the tail (ranks past
/// depth x width) being read only after the attention, at the call's plans, while 4 pool workers share the drive with
/// the call's misses; `whole` reads it during the attention instead, at the cost of the stock schedule's double reads
/// (records the plans evict before any hit). The same admissions, plans, slots and kernels: exact by construction.
/// `early_whole_trim` (X21, ahead-trim-20261010): `early_whole`, then at the plans the read-ahead jobs the reader has not
/// started yet are withdrawn from its queue (they wrote no row) and their records are resolved by the plans as `early`'s
/// tail is (`S.AheadPost.whole_trim`): read once if a group hits them first, never read if a plan evicts them first. At
/// the next9 clock (~76 ms to the plans on a sliding layer) the reader is still on the head: the tail (ranks 96+) is
/// queued and its evicted-first records, ~254-263 per 16K pass = ~3.5 GB, are not read; at the clamped clock (~120 ms)
/// every job has started and the schedule is `early_whole`'s. The same admissions, plans, slots and kernels: exact by
/// construction.
/// X20 `auto` (mimo/sched-auto): `early` below a prompt-length threshold, `early_whole` at or above it, chosen once per
/// request at its start (`AutoSchedule`, `StreamSource.chooseReadSchedule`); the stream is built once for both.
pub const ReadSchedule = enum { current, early, early_whole, early_whole_trim, auto, invalid };

/// X20: `auto`'s default threshold (settings `prefill_read_schedule_threshold`): shortprompt1-20261010 measured `early`
/// ahead of `early_whole` on every short prompt (codectx-4k 341.2/341.4 vs 316.3/318.0, codectx-1k 90.6/90.5 vs
/// 85.6/86.8, std-1k 89.1/89.4 vs 85.6/85.8 prefill tok/s) and P7 try 3 measured `early_whole` +7 % over `early` at 16K.
pub const default_read_schedule_threshold: u32 = 8192;
/// X20: the largest threshold a setting may name (the checkpoint's `max_position_embeddings`, 1,048,576).
pub const max_read_schedule_threshold: u32 = 1 << 20;

/// X20 `auto`: the request's schedule by its prompt length, a construction-time table of two exact routes that share
/// everything the stream is built with (the overlap schedule, its ticket ring, the stock admissions, the route triple)
/// and differ only in the read-ahead posted with the attention (`S.AheadPost`): `below` (`early`, the head) for a
/// prompt of fewer than `threshold` tokens, `at_or_above` (`early_whole`, the whole) otherwise. Consulted once per
/// request, at its start, never per layer or call.
pub const AutoSchedule = struct {
    threshold: u32 = default_read_schedule_threshold,

    pub const below: ReadSchedule = .early;
    pub const at_or_above: ReadSchedule = .early_whole;

    pub fn of(t: AutoSchedule, prompt_tokens: u64) ReadSchedule {
        return if (prompt_tokens < t.threshold) below else at_or_above;
    }
};

/// The `early` schedule's ticket ring: its posts run up to the read-ahead (<= 248 records) plus a whole call's loads
/// (<= 256) ahead of their consumers, 2 tickets each; the stock ring (`io_tickets`) is sized for one group's.
pub const io_tickets_early: u32 = 2048;

pub const applyReadSchedule = Mimo.applyReadSchedule;

const aheadPost = Mimo.aheadPost;

pub const eventOptions = Mimo.eventOptions;

test "mimo event gates: installed stream arms the event with explicit lookahead" {
    const opts = eventOptions(&.{8}, .host, .{ .id = 7, .object = 123 });
    try std.testing.expectEqual(@as(u64, 123), opts.event.?.backend.metal);
    try std.testing.expectEqual(lookahead, opts.lookahead.?);
    try std.testing.expectEqual(@as(u32, 4), opts.pool.workers);
    try std.testing.expectEqual(@as(u8, 2), opts.wide_depth);
}

const wavesOf = Mimo.wavesOf;

pub const traceBank = Mimo.traceBank;

pub const StreamSource = Mimo.StreamSource;

test "mimo sdk2 mxfp4 SDK source: contract, absolute layers, and U32/U8 trace bindings" {
    sdk_ext.expert.assertSource(StreamSource);
    const testing = std.testing;
    const geom = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    var layers: [47]bank_mod.Layer = undefined;
    for (&layers, 0..) |*layer, i| layer.* = .{
        .logical_bytes = geom.record_bytes,
        .base_offset = i * geom.num_experts * geom.record_bytes,
        .segments = geom.segments,
    };
    const bank: bank_mod.SDKBank = .{
        .allocator = testing.allocator,
        .layers = &layers,
        .n_experts = geom.num_experts,
        .first_moe_layer = geom.first_moe_layer,
        .sidecar = .{ .fd = -1, .size = geom.total_bytes },
    };
    // Only metadata is used here: no stream/pool startup or MLX allocation.
    var stream: Stream = undefined;
    stream.bank = &bank;
    var source = StreamSource.init(&stream);
    try testing.expectEqual(@as(u32, 0), source.relativeLayer(1));
    try testing.expectEqual(@as(u32, 46), source.relativeLayer(47));
    try testing.expectEqual(@as(u32, 0), source.bankRows(0, .base));
    var g = @import("../ops.zig").TraceOps.init(testing.allocator);
    defer g.deinit();
    try testing.expect(try source.bankArrays(&g, 0, .transient) == null);
    const arrays = try traceBank(&g, &layers[46], 8);
    try testing.expectEqual(mlx.mlx_dtype.uint32, g.dtypeOf(arrays.gate.w));
    try testing.expectEqual(mlx.mlx_dtype.uint8, g.dtypeOf(arrays.gate.scales.?));
    try testing.expectEqualSlices(c_int, &.{ 8, 2048, 512 }, g.shapeOf(arrays.gate.w).slice());
    try testing.expectEqualSlices(c_int, &.{ 8, 4096, 64 }, g.shapeOf(arrays.down.scales.?).slice());
    const opt = options(&.{8}, .host);
    try testing.expectEqual(@as(u8, 2), opt.wide_depth);
    try testing.expectEqual(@as(u64, 1_283_457_024), opt.transient_rows * geom.record_bytes);
    try testing.expectEqual(@as(u32, 2), opt.records_per_part);
    try testing.expectEqual(@as(u32, 512), opt.pool.tickets);
}

// Synthetic payload only. The parent owns the real GPU guard and service
// lifecycle. Initial slot payload: 18 * 1,632 B; unfilled grow adds 3,264 B.
test "mimo sdk2 mxfp4 SDK GPU: asynchronous windows and bound six-component slot reuse" {
    _ = env.get(.MIMO_Q4_SDK_STREAM_GPU) orelse return error.SkipZigTest;
    const guard = env.get(._GPU_WINDOW_LOCKED) orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const testing = std.testing;
    const a = testing.allocator;
    const geom = try bank_mod.Geometry.init(32, 32, 64, 1, 2);
    var bytes: [2 * 64 * 1632]u8 = undefined;
    for (0..2) |layer| for (0..64) |expert| {
        for (geom.segments, 0..) |seg, component| {
            const value: u8 = @intCast(if (seg.dtype == .U8)
                90 + (layer * 64 + expert + component) % 70
            else
                1 + (layer * 17 + expert + component * 3) % 127);
            const offset = geom.recordOffset(@intCast(layer + 1), @intCast(expert)) + seg.offset;
            @memset(bytes[offset..][0..seg.length], value);
        }
    };
    var side: bank_mod.TestSidecar = undefined;
    try side.init(a, geom, &bytes);
    defer side.deinit();
    const mlx_stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(mlx_stream);
    var g = try @import("../ops.zig").MlxOps.init(a, mlx_stream);
    defer g.deinit();
    const stream = try Stream.init(a, &side.bank, .{
        .rows = &.{ 1, 1 },
        .max_route_ids = 8,
        .transient_rows = 16,
        .records_per_part = records_per_part,
        .wide_depth = wide_depth,
        .pool = .{ .workers = io_workers, .staging_bytes = io_staging_bytes, .tickets = io_tickets },
        .slot_memory = .{ .mlx = mlx_stream },
        .grow_fill = .unfilled,
    });
    defer stream.deinit();
    var source = StreamSource.init(stream);
    var banks = try fixtureBindings(&source, &g);
    try source.seedPrefill(1, &.{ 0, 1, 1, 2 });
    const first_ids = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const second_ids = [_]u16{ 8, 9, 10, 11, 12, 13, 14, 15 };
    const first = try source.route(1, &first_ids, &.{});
    const second = try source.route(1, &second_ids, &.{});
    try testing.expect(first.route != second.route);
    try testing.expect(first.route.?.state == .live and second.route.?.state == .live);
    try testing.expect(first.route.?.window != second.route.?.window);
    try source.holdBase(first);
    try checkFixtureCall(&source, &g, first, &first_ids, &banks[0], &bytes);
    try checkFixtureCall(&source, &g, second, &second_ids, &banks[0], &bytes);
    source.release(first);
    source.release(second);
    try source.flush();
    try testing.expect(stream.held_base.items.len > 0);
    source.releaseHeld();
    try source.grow(&.{ 2, 2 });
    banks = try fixtureBindings(&source, &g);
    try testing.expectEqual(@as(u32, 2), source.slotCapacity(1));
    // More than one 256-ticket ring, alternating layers to overwrite shared
    // transient rows. Bindings remain the same through every overwrite.
    for (0..33) |cycle| {
        const layer: u32 = @intCast(1 + cycle % 2);
        var ids: [8]u16 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast((cycle * 7 + i) % 64);
        const call = try source.route(layer, &ids, &.{});
        for (call.route.?.partsOf()) |part| try testing.expect(part.n <= records_per_part);
        try checkFixtureCall(&source, &g, call, &ids, &banks[layer - 1], &bytes);
        source.release(call);
        try source.flush();
    }
    try testing.expect(source.stats().expert_bytes_read > 128 * geom.record_bytes);
}

fn fixtureBindings(source: *StreamSource, g: *@import("../ops.zig").MlxOps) ![2][3]?BankArrays {
    var banks: [2][3]?BankArrays = undefined;
    for (&banks, 1..) |*layer, absolute| for ([_]BankKind{ .base, .ext, .transient }, 0..) |kind, i| {
        layer[i] = try source.bankArrays(g, @intCast(absolute), kind);
    };
    return banks;
}

fn checkFixtureCall(source: *StreamSource, g: *@import("../ops.zig").MlxOps, call: *StreamSource.Call, ids: []const u16, banks: *const [3]?BankArrays, bytes: []const u8) !void {
    const testing = std.testing;
    const r = call.route.?;
    const layer = &source.stream.bank.layers[r.layer];
    const sv = source.served(call);
    try testing.expectEqual(ids.len, sv.refs.len);
    for (0..sv.n_parts) |part| {
        try source.waitGu(call, @intCast(part));
        try source.waitDown(call, @intCast(part));
    }
    for (ids, r.plan.slotsOf()) |id, slot| {
        const offset = source.stream.bank.recordOffset(r.layer, id);
        for (layer.segments, 0..) |seg, component| {
            try testing.expectEqualSlices(u8, bytes[offset + seg.offset ..][0..seg.length], source.stream.slotRow(r.layer, slot, @fromBackingInt(@intCast(component))));
        }
    }
    for ([_]usize{ 0, ids.len - 1 }) |i| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const ref = sv.refs[i];
        const b = banks[@backingInt(ref.bank)].?;
        const leaf = [_]mlx.mlx_array{ b.gate.w, b.gate.scales.?, b.up.w, b.up.scales.?, b.down.w, b.down.scales.? };
        const row = [_]u32{ref.row};
        const index = try g.hostArray(std.mem.sliceAsBytes(&row), &.{1}, .uint32);
        var gathered: [bank_mod.n_components]mlx.mlx_array = undefined;
        for (&gathered, leaf) |*out, array| out.* = try g.take(array, index, 0);
        try g.evalAll(&gathered);
        try mlx.check(mlx.mlx_synchronize(g.s));
        const offset = source.stream.bank.recordOffset(r.layer, ids[i]);
        for (gathered, layer.segments) |array, seg| {
            const data = mlx.mlx_array_data_uint8(array) orelse return error.MlxNoData;
            try testing.expectEqualSlices(u8, bytes[offset + seg.offset ..][0..seg.length], data[0..seg.length]);
        }
    }
}

// CPU (host slot memory, synthetic six-component records): the phase-rows route's boundary as Source drives it
// (`Source.beginDecode` / `resumePrefill`): the transient release, the grow by decode - prompt rows on every layer
// (window 0 re-created at `decodeTransientRows`), and the reverse change back to the prompt's rows and scratch.
test "mimo phase rows: the boundary frees the scratch, grows every layer by decode - prompt rows, and shrinks back (host)" {
    try phaseRowsBoundary(.{ 2, 2 }, .{ 5, 5 });
}

// mimo/layer-rows: an `expert_rows_by_layer` split (offsets -1, +1 at 2 / 5 rows): each layer holds its own prompt and
// decode rows, grows by the same 3 rows at the boundary, and shrinks back to its own prompt rows.
test "mimo layer rows: a per-layer split holds each layer's own rows through the phase change and back (host)" {
    try phaseRowsBoundary(.{ 1, 3 }, .{ 4, 6 });
}

fn phaseRowsBoundary(prompt_rows: [2]u32, decode_rows: [2]u32) !void {
    const testing = std.testing;
    const a = testing.allocator;
    const geom = try bank_mod.Geometry.init(32, 32, 64, 1, 2);
    var bytes: [2 * 64 * 1632]u8 = @splat(7);
    var side: bank_mod.TestSidecar = undefined;
    try side.init(a, geom, &bytes);
    defer side.deinit();
    var opts = options(&prompt_rows, .host);
    opts.max_route_ids = 8;
    opts.transient_rows = 16;
    opts.transient_release = true; // the route `Source.initBank` installs with decode rows
    const stream = try Stream.init(a, &side.bank, opts);
    defer stream.deinit();
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.releaseTransient());
    try stream.grow(&decode_rows);
    for (stream.layers, decode_rows, prompt_rows) |*ls, d, p| {
        // The growth request: decode - prompt rows per layer, the residents' base rows unmoved.
        try testing.expectEqual(d - p, ls.ext.?.rows);
        try testing.expectEqual(p, ls.base.rows);
        try testing.expectEqual(d, ls.policy.capacity);
    }
    try testing.expectEqual(opts.max_route_ids + S.decode_staging_rows, stream.transient.rows);
    try testing.expectEqual(@as(u32, max_route_ids), decodeTransientRows(.route));
    try testing.expectEqual(@as(u32, 32), decodeTransientRows(.verify));
    // The reverse change frees the grown rows and window 0, then the prompt's scratch comes back.
    var grown: u64 = 0;
    for (decode_rows, prompt_rows) |d, p| grown += d - p;
    try testing.expectEqual(@as(u64, (grown + 8) * geom.record_bytes), try stream.shrink(&prompt_rows, .forget));
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.regrowTransient());
    for (stream.layers, prompt_rows) |*ls, p| {
        try testing.expect(ls.ext == null);
        try testing.expectEqual(p, ls.policy.capacity);
    }
    try testing.expectEqual(@as(u32, 16), stream.transient.rows);
}

// CPU (host slot memory, synthetic records, no MLX): decode's window 0 at `Options.decode_route_ids` rows. The grow
// allocates exactly that many, every decode route's transient loads land inside it (routes of decode_route_ids distinct
// ids, all misses, on layers whose persistent rows fill up: the window's worst case), the shrink frees exactly it, and
// the SDK refuses a decode window wider than the widest route.
test "mimo rows audit R3: decode's window 0 holds decode_route_ids rows and every decode route's transient loads land in it; R2's cancelled growth restores the prompt scratch (host)" {
    const testing = std.testing;
    const a = testing.allocator;
    const geom = try bank_mod.Geometry.init(32, 32, 64, 1, 2);
    const bytes = try a.alloc(u8, geom.total_bytes);
    defer a.free(bytes);
    @memset(bytes, 5);
    var side: bank_mod.TestSidecar = undefined;
    try side.init(a, geom, bytes);
    defer side.deinit();
    const prompt_rows = [_]u32{ 2, 2 };
    const decode_rows = [_]u32{ 3, 3 };
    const window: u32 = 4;
    var opts = options(&prompt_rows, .host);
    opts.max_route_ids = 8;
    opts.transient_rows = 16;
    opts.transient_release = true;
    // Wider than the widest route: refused by name at construction.
    opts.decode_route_ids = 9;
    try testing.expectError(error.InvalidOptions, Stream.init(a, &side.bank, opts));
    opts.decode_route_ids = window;
    const stream = try Stream.init(a, &side.bank, opts);
    defer stream.deinit();
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.releaseTransient());
    var ss = StreamSource.init(stream);
    try ss.grow(&decode_rows);
    try testing.expectEqual(window + S.decode_staging_rows, stream.transient.rows);
    // Decode routes of `window` distinct ids, every one a miss (fresh ids each call), on both routed layers.
    var transient_loads: u32 = 0;
    var next_id: u16 = 0;
    for (0..6) |call_i| {
        const layer: u32 = 1 + @as(u32, @intCast(call_i % 2));
        var ids: [4]u16 = undefined;
        for (&ids) |*e| {
            e.* = next_id;
            next_id += 1;
        }
        const call = try ss.route(layer, &ids, &.{});
        const plan = &call.route.?.plan;
        const capacity = stream.layers[ss.relativeLayer(layer)].policy.capacity;
        try testing.expectEqual(@as(u32, window), plan.n_misses);
        for (plan.loadsOf()) |l| {
            if (l.persistent) {
                try testing.expect(l.slot < capacity);
            } else {
                try testing.expect(l.slot >= capacity and l.slot < capacity + window);
                transient_loads += 1;
            }
        }
        for (0..call.route.?.n_parts) |p| {
            try ss.waitGu(call, @intCast(p));
            try ss.waitDown(call, @intCast(p));
        }
        ss.release(call);
        try ss.flush();
    }
    // A call admits at most its layer's 3 rows: at least one of its 4 misses rode the window every call.
    try testing.expect(transient_loads >= 6);
    // The reverse change frees the grown rows and exactly the narrow window 0.
    try testing.expectEqual(@as(u64, (2 * 1 + window) * geom.record_bytes), try stream.shrink(&prompt_rows, .forget));
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.regrowTransient());
    try testing.expectEqual(@as(u32, 16), stream.transient.rows);
    // R2's cancelled growth (`Source.resumePrefill` with the growth pending: a request that ended before its first decode
    // forward): released, never grown, the prompt's scratch re-created as is; the next prompt routes, and its boundary
    // releases and grows as the first did.
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.releaseTransient());
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.regrowTransient());
    try testing.expect(!ss.decoding());
    var prompt_ids: [8]u16 = undefined;
    for (&prompt_ids, 0..) |*e, i| e.* = @intCast(40 + i);
    {
        const call = try ss.route(1, &prompt_ids, &.{});
        for (0..call.route.?.n_parts) |p| {
            try ss.waitGu(call, @intCast(p));
            try ss.waitDown(call, @intCast(p));
        }
        ss.release(call);
        try ss.flush();
    }
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.releaseTransient());
    try ss.grow(&decode_rows);
    try testing.expect(ss.decoding());
    try testing.expectEqual(window + S.decode_staging_rows, stream.transient.rows);
}

// ── HITPASS CPU tests: a host-memory MiMo stream (layer 0 dense, two routed layers, 64 experts, 4 rows each) over a
// tmp sidecar; pool pread workers and host slot rows only. No MLX array, device or lock. ──

const HitFixture = struct {
    const geom_args = .{ 32, 32, 64, 1, 2 };
    bytes: []u8,
    side: bank_mod.TestSidecar,
    stream: *Stream,

    /// In place: the stream retains `&f.side.bank`, the SDK bank `&f.side.owner`.
    fn init(f: *HitFixture, a: std.mem.Allocator, rows: []const u32) !void {
        return f.initOpts(a, options(rows, .host));
    }

    fn initOpts(f: *HitFixture, a: std.mem.Allocator, opts: Options) !void {
        const geom = try @call(.auto, bank_mod.Geometry.init, geom_args);
        f.bytes = try a.alloc(u8, geom.total_bytes);
        errdefer a.free(f.bytes);
        for (0..2) |layer| for (0..64) |expert| for (geom.segments, 0..) |seg, component| {
            const offset = geom.recordOffset(@intCast(layer + 1), @intCast(expert)) + seg.offset;
            for (f.bytes[offset..][0..seg.length], 0..) |*b, i| b.* = @truncate(layer * 131 + expert * 7 + component * 3 + i);
        };
        try f.side.init(a, geom, f.bytes);
        errdefer f.side.deinit();
        f.stream = try Stream.init(a, &f.side.bank, opts);
    }

    fn deinit(f: *HitFixture, a: std.mem.Allocator) void {
        f.stream.deinit();
        f.side.deinit();
        a.free(f.bytes);
    }
};

// ── P7 read schedule CPU proofs (host stream over the HitFixture's sidecar: two routed layers, 64 experts; pool pread
// workers, host rows; no MLX array, device or lock). The executor's depth-2 wide sequence is emulated: read-ahead, the
// executor's await, seed, hot order, groups 0-1 routed, then per group its reads landed, every routed expert's row
// checked against the sidecar, released, group + 2 routed. ──

/// Groups of 11 over the fixture's 64 experts: at most 6, the stream's route ring (`installReadSchedule`).
const P7Width: u32 = 11;

/// One emulated prompt call's record: per group its plan, window and hit slots (as routed), and (early) how many groups
/// past the first two had every read posted before the executor routed them (the depth schedule posts them only then).
const P7Call = struct {
    plans: [S.route_capacity]expert_policy.Plan = undefined,
    windows: [S.route_capacity]u8 = undefined,
    hit_slots: [S.route_capacity][max_route_ids]u32 = undefined,
    n_groups: u32 = 0,
    posted_ahead: u32 = 0,
};

/// Every component row of `slot` (relative layer `rel`) holds expert `e`'s record bytes from the sidecar.
fn p7ExpectRecord(f: *HitFixture, rel: u32, slot: u32, e: u16) !void {
    const geom = f.side.owner.geometry;
    for (geom.segments, 0..) |seg, c| {
        const want = f.bytes[geom.recordOffset(rel + 1, e) + seg.offset ..][0..seg.length];
        const got = f.stream.slotRow(rel, slot, @fromBackingInt(@intCast(c)))[0..seg.length];
        try std.testing.expectEqualSlices(u8, want, got);
    }
}

/// The emulated executor over one prompt call of absolute layer `layer`: `ids` the call's routed ids (rows x top-k),
/// `predicted` the read-ahead's ranking. Checks every routed expert's bytes when its group is consumed.
fn p7RunCall(f: *HitFixture, ss: *StreamSource, layer: u32, ids: []const u16, predicted: []const u16, out: *P7Call) !void {
    return p7RunCallProbed(f, ss, layer, ids, predicted, out, null);
}

/// `p7RunCall` with `probe` run mid-call: with the read-ahead in flight, and with the call planned (its first route).
fn p7RunCallProbed(f: *HitFixture, ss: *StreamSource, layer: u32, ids: []const u16, predicted: []const u16, out: *P7Call, probe: ?*const fn (*HitFixture, *StreamSource) anyerror!void) !void {
    const rel = layer - 1;
    try ss.readAheadSeed(layer, predicted);
    if (probe) |pr| try pr(f, ss);
    try ss.awaitReadAhead(layer);
    try ss.seedPrefill(layer, ids);
    var count: [64]u32 = @splat(0);
    var distinct_buf: [64]u16 = undefined;
    var nd: usize = 0;
    for (ids) |e| {
        if (count[e] == 0) {
            distinct_buf[nd] = e;
            nd += 1;
        }
        count[e] += 1;
    }
    const distinct = distinct_buf[0..nd];
    ss.hotOrder(layer, distinct, &count);
    const n_groups: u32 = @intCast((nd + P7Width - 1) / P7Width);
    out.n_groups = n_groups;
    const group = struct {
        fn of(d: []const u16, g: u32) []const u16 {
            return d[g * P7Width .. @min((g + 1) * P7Width, d.len)];
        }
    }.of;
    var calls: [2]?*StreamSource.Call = .{ null, null };
    for (0..@min(2, n_groups)) |g| calls[g] = try ss.route(layer, group(distinct, @intCast(g)), &.{});
    if (probe) |pr| try pr(f, ss);
    for (0..n_groups) |gi| {
        const g: u32 = @intCast(gi);
        const call = calls[g % 2].?;
        const r = call.route.?;
        out.plans[g] = r.plan;
        out.windows[g] = r.window;
        @memcpy(out.hit_slots[g][0..r.plan.n_hits], r.hit_slots[0..r.plan.n_hits]);
        for (0..r.n_parts) |pi| {
            try ss.waitGu(call, @intCast(pi));
            try ss.waitDown(call, @intCast(pi));
        }
        // The group's consumers: every routed expert's row holds its record.
        for (group(distinct, g)) |e| {
            const slot = blk: {
                for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |h, hs| if (h == e) break :blk hs;
                for (r.plan.loadsOf()) |l| if (l.expert == e) break :blk l.slot;
                return error.P7TestExpertNotServed;
            };
            try p7ExpectRecord(f, rel, slot, e);
        }
        ss.release(call);
        calls[g % 2] = null;
        if (g + 2 < n_groups) {
            // Issue order: group g + 2's reads already posted (all its parts) before the executor routes it.
            if (ss.early) out.posted_ahead += @intFromBool(ss.stream.p7.post_g > g + 2);
            calls[g % 2] = try ss.route(layer, group(distinct, g + 2), &.{});
        }
    }
    try ss.flush();
}

/// A call's routed ids: `rows` tokens x 8 distinct experts from a skewed draw over `span` experts (seeded).
fn p7Ids(rng: std.Random, rows: usize, span: u16, out: []u16) []u16 {
    var n: usize = 0;
    for (0..rows) |_| {
        var picked: [8]u16 = undefined;
        var k: usize = 0;
        while (k < 8) {
            // Skew: a squared uniform favours low ids (hot experts), so groups differ in reads and hits.
            const u = rng.float(f32);
            const e: u16 = @intFromFloat(@min(@as(f32, @floatFromInt(span - 1)), u * u * @as(f32, @floatFromInt(span))));
            if (std.mem.indexOfScalar(u16, picked[0..k], e) != null) continue;
            picked[k] = e;
            k += 1;
        }
        @memcpy(out[n..][0..8], &picked);
        n += 8;
    }
    return out[0..n];
}

/// The reader a P7 script runs on: its workers, and (X21) what keeps it behind the call's plans: a sleep injected on each
/// call's first read-ahead read (`hold_ns`) or a random delay before every range read (`delay_ns`, at most); `relocate`:
/// the stream built with the overlap schedule's relocation (`Options.prompt_relocate`, glm5_next's route; MiMo's off).
const P7Pool = struct { workers: u32 = io_workers, hold_ns: i64 = 0, delay_ns: i64 = 0, relocate: bool = false };

fn p7Fixture(f: *HitFixture, a: std.mem.Allocator, rows: []const u32, schedule: ReadSchedule) !StreamSource {
    return p7FixtureOn(f, a, rows, schedule, .{});
}

fn p7FixtureOn(f: *HitFixture, a: std.mem.Allocator, rows: []const u32, schedule: ReadSchedule, pool: P7Pool) !StreamSource {
    var opts = options(rows, .host);
    opts.max_route_ids = P7Width;
    opts.transient_rows = wide_depth * P7Width;
    opts.pool.workers = pool.workers;
    try applyReadSchedule(&opts, schedule);
    opts.prompt_relocate = pool.relocate;
    try f.initOpts(a, opts);
    errdefer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    try ss.installReadSchedule(schedule, default_read_schedule_threshold);
    return ss;
}

fn p7ExpectSamePlans(want: *const P7Call, got: *const P7Call) !void {
    const testing = std.testing;
    try testing.expectEqual(want.n_groups, got.n_groups);
    for (0..want.n_groups) |g| {
        const x = &want.plans[g];
        const y = &got.plans[g];
        try testing.expectEqualSlices(u32, x.slotsOf(), y.slotsOf());
        try testing.expectEqualSlices(u16, x.hitsOf(), y.hitsOf());
        try testing.expectEqualSlices(u16, x.missesOf(), y.missesOf());
        try testing.expectEqual(x.n_loads, y.n_loads);
        for (x.loadsOf(), y.loadsOf()) |lx, ly| try testing.expect(lx.expert == ly.expert and lx.slot == ly.slot and lx.persistent == ly.persistent);
        try testing.expectEqual(x.n_evictions, y.n_evictions);
        for (x.evictionsOf(), y.evictionsOf()) |ex, ey| try testing.expect(ex.slot == ey.slot and ex.previous == ey.previous and ex.next == ey.next);
        try testing.expectEqual(want.windows[g], got.windows[g]);
        try testing.expectEqualSlices(u32, want.hit_slots[g][0..x.n_hits], got.hit_slots[g][0..y.n_hits]);
    }
}

/// One schedule's run over a fixed list of prompt calls (the pool is one per process: the two schedules run in turn).
const P7Run = struct {
    calls: []P7Call,
    stats: Stats,
    residents: [2][64]u16,
    /// X20: per request, the schedule its start chose (`chooseReadSchedule`) and its own bytes read and records dropped.
    chosen: [p7_max_requests]ReadSchedule = undefined,
    req_bytes: [p7_max_requests]u64 = undefined,
    req_dropped: [p7_max_requests]u64 = undefined,
    n_requests: usize = 0,
};
const p7_max_requests = 16;

/// The call list: per prompt, per routed layer, its routed ids and the read-ahead's ranking.
const P7Script = struct {
    ids: [][]const u16,
    pred: [][]const u16,
    /// A prompt boundary (residents forgotten, prompt counts zeroed) after every `per_prompt` calls.
    per_prompt: usize,
    /// X20: request r's prompt length (`prompt_tokens[r % len]`), handed to `chooseReadSchedule` at its start: `auto`
    /// chooses by it, a named schedule ignores it.
    prompt_tokens: []const u64 = &.{16_384},
};

fn p7RunScript(a: std.mem.Allocator, schedule: ReadSchedule, rows: []const u32, script: P7Script, out: *P7Run) !void {
    return p7RunScriptOn(a, schedule, rows, script, out, .{});
}

/// X21: an injected sleep on the call's first read-ahead read (the first predicted record not resident: the read-ahead's
/// first admission, its gate/up range's aligned offset), so a one-worker reader is still on the read-ahead's first job
/// when the call's plans exist and every later job is queued.
fn p7HoldFirstRead(f: *HitFixture, rel: u32, pred: []const u16, ns: i64) void {
    const pol = &f.stream.layers[rel].policy;
    for (pred) |e| if (pol.slotOf(e) == null) {
        const page: u64 = std.heap.pageSize();
        sdk_ext.expert.io.injectFault(f.stream.bank.spans(rel, e).gu_offset / page * page, 5, ns);
        return;
    };
}

fn p7RunScriptOn(a: std.mem.Allocator, schedule: ReadSchedule, rows: []const u32, script: P7Script, out: *P7Run, pool: P7Pool) !void {
    var f: HitFixture = undefined;
    var ss = try p7FixtureOn(&f, a, rows, schedule, pool);
    defer f.deinit(a);
    const inject = sdk_ext.expert.io.test_abi.abi;
    if (pool.delay_ns > 0) inject.q3ld_test_delay(0x7e7a11, pool.delay_ns);
    defer if (pool.delay_ns > 0) inject.q3ld_test_delay(0, 0);
    defer if (pool.hold_ns > 0) sdk_ext.expert.io.clearFaults();
    out.n_requests = 0;
    var at_start: Stats = .{};
    for (script.ids, script.pred, 0..) |ids, pred, i| {
        if (i % script.per_prompt == 0) {
            // X20: a request's start (nothing live: construction, or the boundary below): its schedule, once.
            const r = out.n_requests;
            if (r == p7_max_requests) return error.P7TestTooManyRequests;
            out.chosen[r] = try ss.chooseReadSchedule(script.prompt_tokens[r % script.prompt_tokens.len]);
            out.n_requests += 1;
            at_start = ss.stats();
        }
        const layer: u32 = @intCast(1 + i % 2);
        if (pool.hold_ns > 0) p7HoldFirstRead(&f, layer - 1, pred, pool.hold_ns);
        out.calls[i] = .{};
        try p7RunCall(&f, &ss, layer, ids, pred, &out.calls[i]);
        // X20: every call of a request runs the post its start chose (an overlap stream; a stock stream has no post).
        if (f.stream.prompt_reads == .overlap) try std.testing.expectEqual(aheadPost(out.chosen[out.n_requests - 1]), f.stream.prompt_ahead);
        if ((i + 1) % script.per_prompt == 0) {
            // A request's start as `Source.beginPrefill` makes it: routes and the read-ahead settled, then forgotten.
            try ss.stream.settleRoutes();
            _ = try ss.forgetResidents();
            ss.resetPromptCounts();
            const now = ss.stats();
            out.req_bytes[out.n_requests - 1] = now.expert_bytes_read - at_start.expert_bytes_read;
            out.req_dropped[out.n_requests - 1] = now.ahead_dropped - at_start.ahead_dropped;
        }
    }
    out.stats = ss.stats();
    for (f.stream.layers, &out.residents) |*ls, *res| @memcpy(res, ls.policy.slot_to_expert[0..64]);
}

/// The first P7 proof's 40 calls (seeded 0x7e7a11, rows 28): calls alternate layers 1 and 2, expert spans cycle over
/// 24..44 (3-4 groups of 11, under the fixture's 64); the read-ahead's ranking is the call's true hot order with
/// neighbours swapped, then (odd calls) a few cold experts pulled into the admitted ranks, head or tail (mispredictions:
/// drops, read-ahead rows evicted, seed misses).
const P7Calls40 = struct {
    const n_calls = 40;
    ids_store: [n_calls][64 * 8]u16 = undefined,
    pred_store: [n_calls][64]u16 = undefined,
    ids: [n_calls][]const u16 = undefined,
    pred: [n_calls][]const u16 = undefined,

    /// In place (the slices point into the stores).
    fn fill(c: *P7Calls40, rows: u32) void {
        var prng = std.Random.DefaultPrng.init(0x7e7a11);
        const rng = prng.random();
        for (0..n_calls) |i| {
            const span: u16 = @intCast(24 + 4 * ((i / 2) % 6));
            const x = p7Ids(rng, 40, span, &c.ids_store[i]);
            c.ids[i] = x;
            var count: [64]u32 = @splat(0);
            for (x) |e| count[e] += 1;
            var ranked: [64]u16 = undefined;
            const hot = expert_policy.rankHottest(&count, &ranked);
            @memcpy(c.pred_store[i][0..hot.len], hot);
            var k: usize = 1;
            while (k + 1 < hot.len) : (k += 3) std.mem.swap(u16, &c.pred_store[i][k], &c.pred_store[i][k + 1]);
            if (i % 2 == 1 and hot.len > rows) for (0..4) |_| {
                const a_pos = rng.uintLessThan(usize, rows);
                const b_pos = rows + rng.uintLessThan(usize, hot.len - rows);
                std.mem.swap(u16, &c.pred_store[i][a_pos], &c.pred_store[i][b_pos]);
            };
            c.pred[i] = c.pred_store[i][0..hot.len];
        }
    }
};

test "mimo p7 read schedule CPU: early plans every call exactly as current does, every routed row holds its record when its group runs, and reads no more" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    const n_calls = P7Calls40.n_calls;
    var c40: P7Calls40 = .{};
    c40.fill(rows[0]);
    // A prompt per 4 calls: a layer's second call routes against the residency its first left.
    const script: P7Script = .{ .ids = &c40.ids, .pred = &c40.pred, .per_prompt = 4 };
    var calls_c: [n_calls]P7Call = undefined;
    var calls_e: [n_calls]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var early: P7Run = .{ .calls = &calls_e, .stats = .{}, .residents = undefined };
    var calls_w: [n_calls]P7Call = undefined;
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early, &rows, script, &early);
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    var posted_ahead: u32 = 0;
    for (&calls_c, &calls_e, &calls_w) |*want, *got, *got_w| {
        try p7ExpectSamePlans(want, got);
        try p7ExpectSamePlans(want, got_w);
        posted_ahead += got.posted_ahead;
    }
    // The residency every schedule leaves is the same; early reads no more than current, and posted ahead; early_whole
    // reads exactly what current reads (the whole read-ahead, nothing dropped), only earlier.
    for (cur.residents, early.residents, whole.residents) |rc, re, rw| {
        try testing.expectEqualSlices(u16, &rc, &re);
        try testing.expectEqualSlices(u16, &rc, &rw);
    }
    try testing.expect(early.stats.expert_bytes_read <= cur.stats.expert_bytes_read);
    try testing.expectEqual(cur.stats.expert_bytes_read, whole.stats.expert_bytes_read);
    try testing.expectEqual(@as(u64, 0), whole.stats.ahead_dropped);
    for ([_]*const P7Run{ &early, &whole }) |run| {
        try testing.expectEqual(cur.stats.expert_cache_hits, run.stats.expert_cache_hits);
        try testing.expectEqual(cur.stats.expert_cache_misses, run.stats.expert_cache_misses);
    }
    try testing.expect(posted_ahead > 0);
    std.debug.print("MIMO_P7_SCHEDULE calls={d} bytes current={d} early={d} early_whole={d} dropped={d} posted_ahead={d}\n", .{ n_calls, cur.stats.expert_bytes_read, early.stats.expert_bytes_read, whole.stats.expert_bytes_read, early.stats.ahead_dropped, posted_ahead });
}

test "mimo p7 read schedule CPU: a read-ahead record the plans evict before any hit is dropped (read once, not twice)" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    // 40 experts, each on 8 of 40 rows (8 per row): equal counts, so the hot order is the id order (ties by id).
    var ids_buf: [64 * 8]u16 = undefined;
    var n: usize = 0;
    for (0..40) |row| {
        for (0..8) |k| {
            ids_buf[n] = @intCast((row + k * 5) % 40);
            n += 1;
        }
    }
    var count: [64]u32 = @splat(0);
    for (ids_buf[0..n]) |e| count[e] += 1;
    var ranked: [64]u16 = undefined;
    const hot = expert_policy.rankHottest(&count, &ranked);
    // The prediction: the true top 22 (the head: 2 groups of 11), then six cold experts (true ranks 34-39) in the tail
    // instead of true ranks 22-27: unprotected by the seed, evicted by its misses (group 2) before their own group (3).
    var pred_buf: [64]u16 = undefined;
    @memcpy(pred_buf[0..22], hot[0..22]);
    @memcpy(pred_buf[22..28], hot[34..40]);
    @memcpy(pred_buf[28..40], hot[22..34]);
    var ids = [_][]const u16{ids_buf[0..n]};
    var pred = [_][]const u16{pred_buf[0..40]};
    const script: P7Script = .{ .ids = &ids, .pred = &pred, .per_prompt = 1 };
    var calls_c: [1]P7Call = undefined;
    var calls_e: [1]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var early: P7Run = .{ .calls = &calls_e, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early, &rows, script, &early);
    try p7ExpectSamePlans(&calls_c[0], &calls_e[0]);
    // The six cold tail records: evicted by group 2's seed misses before group 3 routes them; current reads them twice.
    try testing.expectEqual(@as(u64, 6), early.stats.ahead_dropped);
    try testing.expectEqual(@as(u64, 0), cur.stats.ahead_dropped);
    try testing.expect(early.stats.expert_bytes_read < cur.stats.expert_bytes_read);
    // early_whole posts the whole read-ahead with the attention: the same plans, nothing dropped, current's bytes.
    var calls_w: [1]P7Call = undefined;
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    try p7ExpectSamePlans(&calls_c[0], &calls_w[0]);
    try testing.expectEqual(@as(u64, 0), whole.stats.ahead_dropped);
    try testing.expectEqual(cur.stats.expert_bytes_read, whole.stats.expert_bytes_read);
}

/// The crafted call of the "dropped" test: 40 experts on 40 rows (equal counts: the hot order is the id order), and a
/// read-ahead ranking whose ranks 22-27 are six cold experts (true ranks 34-39) that group 2's misses evict before group 3
/// routes them.
fn p7CraftedCall(ids_buf: *[64 * 8]u16, pred_buf: *[64]u16) struct { ids: []const u16, pred: []const u16 } {
    var n: usize = 0;
    for (0..40) |row| {
        for (0..8) |k| {
            ids_buf[n] = @intCast((row + k * 5) % 40);
            n += 1;
        }
    }
    var count: [64]u32 = @splat(0);
    for (ids_buf[0..n]) |e| count[e] += 1;
    var ranked: [64]u16 = undefined;
    const hot = expert_policy.rankHottest(&count, &ranked);
    @memcpy(pred_buf[0..22], hot[0..22]);
    @memcpy(pred_buf[22..28], hot[34..40]);
    @memcpy(pred_buf[28..40], hot[22..34]);
    return .{ .ids = ids_buf[0..n], .pred = pred_buf[0..40] };
}

test "mimo p7 read schedule CPU X21: early_whole_trim withdraws the read-ahead jobs still queued at the plans and never reads the evicted-first records among them: early's dropped set, current's plans, hits, misses and rows" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var ids_buf: [64 * 8]u16 = undefined;
    var pred_buf: [64]u16 = undefined;
    const call = p7CraftedCall(&ids_buf, &pred_buf);
    var ids = [_][]const u16{call.ids};
    var pred = [_][]const u16{call.pred};
    const script: P7Script = .{ .ids = &ids, .pred = &pred, .per_prompt = 1 };
    var calls_c: [1]P7Call = undefined;
    var calls_e: [1]P7Call = undefined;
    var calls_w: [1]P7Call = undefined;
    var calls_t: [1]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var early: P7Run = .{ .calls = &calls_e, .stats = .{}, .residents = undefined };
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    var trim: P7Run = .{ .calls = &calls_t, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early, &rows, script, &early);
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    // One reader worker held 300 ms on the read-ahead's first read: at the plans job 0 (ranks 0-7) runs and jobs 1-3
    // (ranks 8-27) are queued. The trim withdraws them; the plans keep ranks 8-21 (groups 0 and 1 hit them first) and
    // drop the six cold records (ranks 22-27: group 2 evicts them before group 3 routes them).
    try p7RunScriptOn(a, .early_whole_trim, &rows, script, &trim, .{ .workers = 1, .hold_ns = 300 * std.time.ns_per_ms });
    try p7ExpectSamePlans(&calls_c[0], &calls_t[0]);
    for (cur.residents, trim.residents) |rc, rt| try testing.expectEqualSlices(u16, &rc, &rt);
    try testing.expectEqual(cur.stats.expert_cache_hits, trim.stats.expert_cache_hits);
    try testing.expectEqual(cur.stats.expert_cache_misses, trim.stats.expert_cache_misses);
    // The withdrawn records the plans evict first are exactly early's dropped ones: the same six, the same bytes.
    try testing.expectEqual(@as(u64, 6), trim.stats.ahead_dropped);
    try testing.expectEqual(early.stats.ahead_dropped, trim.stats.ahead_dropped);
    try testing.expectEqual(early.stats.expert_bytes_read, trim.stats.expert_bytes_read);
    try testing.expectEqual(early.stats.ahead_posted, trim.stats.ahead_posted);
    // early_whole reads them twice (current's bytes); the trim reads each record once.
    try testing.expectEqual(cur.stats.expert_bytes_read, whole.stats.expert_bytes_read);
    try testing.expect(trim.stats.expert_bytes_read < whole.stats.expert_bytes_read);
    try testing.expectEqual(whole.stats.ahead_posted - 6, trim.stats.ahead_posted);
    std.debug.print("MIMO_P7_TRIM crafted bytes current={d} early={d} early_whole={d} early_whole_trim={d} dropped early={d} trim={d} posted whole={d} trim={d}\n", .{ cur.stats.expert_bytes_read, early.stats.expert_bytes_read, whole.stats.expert_bytes_read, trim.stats.expert_bytes_read, early.stats.ahead_dropped, trim.stats.ahead_dropped, whole.stats.ahead_posted, trim.stats.ahead_posted });
}

test "mimo p7 read schedule CPU X21: early_whole_trim under any reader timing plans every call as current does, every routed row holds its record when its group runs, and reads no more than early_whole" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var prng = std.Random.DefaultPrng.init(0x7e7a11);
    const rng = prng.random();
    const n_calls = 40;
    var ids_store: [n_calls][64 * 8]u16 = undefined;
    var pred_store: [n_calls][64]u16 = undefined;
    var ids: [n_calls][]const u16 = undefined;
    var pred: [n_calls][]const u16 = undefined;
    for (0..n_calls) |i| {
        // The first P7 test's script: spans 24..44, neighbours swapped, odd calls with four cold experts pulled into the
        // admitted ranks.
        const span: u16 = @intCast(24 + 4 * ((i / 2) % 6));
        const x = p7Ids(rng, 40, span, &ids_store[i]);
        ids[i] = x;
        var count: [64]u32 = @splat(0);
        for (x) |e| count[e] += 1;
        var ranked: [64]u16 = undefined;
        const hot = expert_policy.rankHottest(&count, &ranked);
        @memcpy(pred_store[i][0..hot.len], hot);
        var k: usize = 1;
        while (k + 1 < hot.len) : (k += 3) std.mem.swap(u16, &pred_store[i][k], &pred_store[i][k + 1]);
        if (i % 2 == 1 and hot.len > rows[0]) for (0..4) |_| {
            const a_pos = rng.uintLessThan(usize, rows[0]);
            const b_pos = rows[0] + rng.uintLessThan(usize, hot.len - rows[0]);
            std.mem.swap(u16, &pred_store[i][a_pos], &pred_store[i][b_pos]);
        };
        pred[i] = pred_store[i][0..hot.len];
    }
    const script: P7Script = .{ .ids = &ids, .pred = &pred, .per_prompt = 4 };
    var calls_c: [n_calls]P7Call = undefined;
    var calls_e: [n_calls]P7Call = undefined;
    var calls_w: [n_calls]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var early: P7Run = .{ .calls = &calls_e, .stats = .{}, .residents = undefined };
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early, &rows, script, &early);
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    // Three reader timings: the stock four workers (most jobs taken before the plans), one worker with a random delay of
    // up to 2 ms before every range (jobs 1+ queued at the plans), one worker held on each call's first read.
    const pools = [_]P7Pool{ .{}, .{ .workers = 1, .delay_ns = 2 * std.time.ns_per_ms }, .{ .workers = 1, .hold_ns = 50 * std.time.ns_per_ms } };
    for (pools) |pool| {
        var calls_t: [n_calls]P7Call = undefined;
        var trim: P7Run = .{ .calls = &calls_t, .stats = .{}, .residents = undefined };
        try p7RunScriptOn(a, .early_whole_trim, &rows, script, &trim, pool);
        for (&calls_c, &calls_t) |*want, *got| try p7ExpectSamePlans(want, got);
        for (cur.residents, trim.residents) |rc, rt| try testing.expectEqualSlices(u16, &rc, &rt);
        try testing.expectEqual(cur.stats.expert_cache_hits, trim.stats.expert_cache_hits);
        try testing.expectEqual(cur.stats.expert_cache_misses, trim.stats.expert_cache_misses);
        try testing.expect(trim.stats.expert_bytes_read <= whole.stats.expert_bytes_read);
        // Each dropped record is one record's bytes less than early_whole; nothing else changes the reads.
        const record = f_record: {
            const geom = try @call(.auto, bank_mod.Geometry.init, HitFixture.geom_args);
            break :f_record geom.record_bytes;
        };
        try testing.expectEqual(whole.stats.expert_bytes_read - trim.stats.ahead_dropped * record, trim.stats.expert_bytes_read);
        // A reader kept behind the plans withdraws early's whole tail (ranks 22+ in jobs 2-3) at least.
        if (pool.workers == 1) try testing.expect(trim.stats.ahead_dropped >= early.stats.ahead_dropped);
        std.debug.print("MIMO_P7_TRIM script calls={d} workers={d} delay_ns={d} hold_ns={d} bytes early_whole={d} early_whole_trim={d} early={d} dropped trim={d} early={d}\n", .{ n_calls, pool.workers, pool.delay_ns, pool.hold_ns, whole.stats.expert_bytes_read, trim.stats.expert_bytes_read, early.stats.expert_bytes_read, trim.stats.ahead_dropped, early.stats.ahead_dropped });
    }
}

/// One record's bytes in the P7 fixture's bank.
fn p7RecordBytes() !u64 {
    const geom = try @call(.auto, bank_mod.Geometry.init, HitFixture.geom_args);
    return geom.record_bytes;
}

/// The relocation proofs' second crafted call: 50 experts on 50 rows (equal counts: the hot order is the id order; five
/// groups of 11), and a read-ahead ranking whose ranks 22-27 are group 4's six experts (true ranks 44-49). Group 2's seed
/// misses evict them; group 4 (window 0) loads them into transient rows 0-5, of which group 2 (window 0) loaded rows 0-4
/// first: only true rank 49's row (5) has no earlier reader.
fn p7CraftedCall50(ids_buf: *[64 * 8]u16, pred_buf: *[64]u16) struct { ids: []const u16, pred: []const u16 } {
    var n: usize = 0;
    for (0..50) |row| {
        for (0..8) |k| {
            ids_buf[n] = @intCast((row + k * 5) % 50);
            n += 1;
        }
    }
    var count: [64]u32 = @splat(0);
    for (ids_buf[0..n]) |e| count[e] += 1;
    var ranked: [64]u16 = undefined;
    const hot = expert_policy.rankHottest(&count, &ranked);
    @memcpy(pred_buf[0..22], hot[0..22]);
    @memcpy(pred_buf[22..28], hot[44..50]);
    @memcpy(pred_buf[28..50], hot[22..44]);
    return .{ .ids = ids_buf[0..n], .pred = pred_buf[0..50] };
}

test "mimo p7 read schedule CPU relocate: the read-ahead records the plans evict before any hit are copied into their later transient rows at the plans and read once: current's plans, hits, misses and rows; every routed row holds its record" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    const record = try p7RecordBytes();
    var ids_buf: [64 * 8]u16 = undefined;
    var pred_buf: [64]u16 = undefined;
    const call = p7CraftedCall(&ids_buf, &pred_buf);
    var ids = [_][]const u16{call.ids};
    var pred = [_][]const u16{call.pred};
    const script: P7Script = .{ .ids = &ids, .pred = &pred, .per_prompt = 1 };
    var calls_c: [1]P7Call = undefined;
    var calls_w: [1]P7Call = undefined;
    var calls_r: [1]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    var moved: P7Run = .{ .calls = &calls_r, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    // early_whole reads the whole read-ahead (nothing withdrawn): the six cold records (true ranks 34-39, group 3, window
    // 1, whose rows group 1 never loads) are read ahead, evicted by group 2's seed misses, and copied at the plans into
    // group 3's transient rows: group 3's loads of them hold, and read nothing.
    try p7RunScriptOn(a, .early_whole, &rows, script, &moved, .{ .relocate = true });
    try p7ExpectSamePlans(&calls_c[0], &calls_r[0]);
    for (cur.residents, moved.residents) |rc, rm| try testing.expectEqualSlices(u16, &rc, &rm);
    try testing.expectEqual(cur.stats.expert_cache_hits, moved.stats.expert_cache_hits);
    try testing.expectEqual(cur.stats.expert_cache_misses, moved.stats.expert_cache_misses);
    try testing.expectEqual(whole.stats.loads_skipped + 6, moved.stats.loads_skipped);
    try testing.expectEqual(whole.stats.expert_bytes_read - 6 * record, moved.stats.expert_bytes_read);
    try testing.expectEqual(whole.stats.ahead_posted, moved.stats.ahead_posted);
    try testing.expectEqual(@as(u64, 0), moved.stats.ahead_dropped);
    // early_whole_trim with the reader held on the read-ahead's first read: the trim withdraws the six (never read), so
    // nothing is relocated and the bytes are the trim's.
    var calls_t: [1]P7Call = undefined;
    var trim: P7Run = .{ .calls = &calls_t, .stats = .{}, .residents = undefined };
    try p7RunScriptOn(a, .early_whole_trim, &rows, script, &trim, .{ .workers = 1, .hold_ns = 300 * std.time.ns_per_ms, .relocate = true });
    try p7ExpectSamePlans(&calls_c[0], &calls_t[0]);
    try testing.expectEqual(@as(u64, 6), trim.stats.ahead_dropped);
    try testing.expectEqual(whole.stats.loads_skipped, trim.stats.loads_skipped);
    try testing.expectEqual(moved.stats.expert_bytes_read, trim.stats.expert_bytes_read);
    std.debug.print("MIMO_P7_RELOCATE crafted bytes current={d} early_whole={d} relocate={d} trim_relocate={d} skipped whole={d} relocate={d}\n", .{ cur.stats.expert_bytes_read, whole.stats.expert_bytes_read, moved.stats.expert_bytes_read, trim.stats.expert_bytes_read, whole.stats.loads_skipped, moved.stats.loads_skipped });
}

test "mimo p7 read schedule CPU relocate: a record whose later transient row an earlier group of the call loads is not relocated (read again, as the control)" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    const record = try p7RecordBytes();
    var ids_buf: [64 * 8]u16 = undefined;
    var pred_buf: [64]u16 = undefined;
    const call = p7CraftedCall50(&ids_buf, &pred_buf);
    var ids = [_][]const u16{call.ids};
    var pred = [_][]const u16{call.pred};
    const script: P7Script = .{ .ids = &ids, .pred = &pred, .per_prompt = 1 };
    var calls_c: [1]P7Call = undefined;
    var calls_w: [1]P7Call = undefined;
    var calls_r: [1]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    var moved: P7Run = .{ .calls = &calls_r, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    try p7RunScriptOn(a, .early_whole, &rows, script, &moved, .{ .relocate = true });
    try testing.expectEqual(@as(u32, 5), calls_c[0].n_groups);
    try p7ExpectSamePlans(&calls_c[0], &calls_r[0]);
    for (cur.residents, moved.residents) |rc, rm| try testing.expectEqualSlices(u16, &rc, &rm);
    // Group 4's six loads: transient rows 0-5 of window 0; group 2 loaded rows 0-4 first. One relocation (row 5).
    const g4 = &calls_c[0].plans[4];
    try testing.expectEqual(@as(u32, 6), g4.n_loads);
    try testing.expectEqual(@as(u8, 0), calls_c[0].windows[4]);
    try testing.expectEqual(@as(u64, 1), moved.stats.loads_skipped - whole.stats.loads_skipped);
    try testing.expectEqual(whole.stats.expert_bytes_read - record, moved.stats.expert_bytes_read);
}

test "mimo p7 read schedule CPU relocate: under any reader timing every call plans as current does, every routed row holds its record, and each record withdrawn or relocated is one record's bytes less than early_whole" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    const record = try p7RecordBytes();
    const n_calls = P7Calls40.n_calls;
    var c40: P7Calls40 = .{};
    c40.fill(rows[0]);
    const script: P7Script = .{ .ids = &c40.ids, .pred = &c40.pred, .per_prompt = 4 };
    var calls_c: [n_calls]P7Call = undefined;
    var calls_w: [n_calls]P7Call = undefined;
    var cur: P7Run = .{ .calls = &calls_c, .stats = .{}, .residents = undefined };
    var whole: P7Run = .{ .calls = &calls_w, .stats = .{}, .residents = undefined };
    try p7RunScript(a, .current, &rows, script, &cur);
    try p7RunScript(a, .early_whole, &rows, script, &whole);
    const Arm = struct { schedule: ReadSchedule, pool: P7Pool };
    const arms = [_]Arm{
        .{ .schedule = .early_whole, .pool = .{ .relocate = true } },
        .{ .schedule = .early_whole_trim, .pool = .{ .relocate = true } },
        .{ .schedule = .early_whole_trim, .pool = .{ .workers = 1, .delay_ns = 2 * std.time.ns_per_ms, .relocate = true } },
        .{ .schedule = .early_whole_trim, .pool = .{ .workers = 1, .hold_ns = 50 * std.time.ns_per_ms, .relocate = true } },
    };
    for (arms) |arm| {
        var calls_r: [n_calls]P7Call = undefined;
        var run: P7Run = .{ .calls = &calls_r, .stats = .{}, .residents = undefined };
        try p7RunScriptOn(a, arm.schedule, &rows, script, &run, arm.pool);
        for (&calls_c, &calls_r) |*want, *got| try p7ExpectSamePlans(want, got);
        for (cur.residents, run.residents) |rc, rr| try testing.expectEqualSlices(u16, &rc, &rr);
        try testing.expectEqual(cur.stats.expert_cache_hits, run.stats.expert_cache_hits);
        try testing.expectEqual(cur.stats.expert_cache_misses, run.stats.expert_cache_misses);
        try testing.expect(run.stats.loads_skipped >= whole.stats.loads_skipped);
        const relocated = run.stats.loads_skipped - whole.stats.loads_skipped;
        try testing.expectEqual(whole.stats.expert_bytes_read - (run.stats.ahead_dropped + relocated) * record, run.stats.expert_bytes_read);
        // early_whole withdraws nothing: every evicted-first record is read ahead, and the script's mispredicted calls
        // have ones whose later rows are fresh.
        if (arm.schedule == .early_whole) try testing.expect(relocated > 0 and run.stats.ahead_dropped == 0);
        std.debug.print("MIMO_P7_RELOCATE script calls={d} schedule={s} workers={d} delay_ns={d} hold_ns={d} bytes early_whole={d} run={d} relocated={d} dropped={d}\n", .{ n_calls, @tagName(arm.schedule), arm.pool.workers, arm.pool.delay_ns, arm.pool.hold_ns, whole.stats.expert_bytes_read, run.stats.expert_bytes_read, relocated, run.stats.ahead_dropped });
    }
}

test "mimo p7 read schedule relocate: a stream refuses relocation without the overlap schedule; MiMo's schedules leave it off" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var stock = options(&rows, .host);
    stock.prompt_relocate = true;
    var f: HitFixture = undefined;
    try testing.expectError(error.InvalidOptions, f.initOpts(a, stock));
    for ([_]ReadSchedule{ .current, .early, .early_whole, .early_whole_trim, .auto }) |schedule| {
        var opts = options(&rows, .host);
        try applyReadSchedule(&opts, schedule);
        try testing.expect(!opts.prompt_relocate);
    }
}

test "mimo p7 read schedule: a stream built for early_whole_trim refuses the other schedules; early_whole_trim sets the trimmed whole post" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var f: HitFixture = undefined;
    var ss = try p7Fixture(&f, a, &rows, .early_whole_trim);
    defer f.deinit(a);
    const t = default_read_schedule_threshold;
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.early_whole, t));
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.early, t));
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.current, t));
    try ss.installReadSchedule(.early_whole_trim, t);
    var opts = options(&rows, .host);
    try applyReadSchedule(&opts, .early_whole_trim);
    try testing.expectEqual(S.AheadPost.whole_trim, opts.prompt_ahead);
    try testing.expectEqual(S.PromptReads.overlap, opts.prompt_reads);
    try testing.expectEqual(io_tickets_early, opts.pool.tickets);
    try testing.expectEqual(@as(?u32, (io_tickets - 2 * sdk_ext.expert.io.max_items) / 2), opts.ahead_cap);
    // The trimmed post needs the overlap schedule.
    var stock = options(&rows, .host);
    stock.prompt_ahead = .whole_trim;
    var g: HitFixture = undefined;
    try testing.expectError(error.InvalidOptions, g.initOpts(a, stock));
}

test "mimo p7 read schedule: installation refuses a stream built for the other schedule, and the setting by name" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var f: HitFixture = undefined;
    var ss = try p7Fixture(&f, a, &rows, .current);
    defer f.deinit(a);
    const t = default_read_schedule_threshold;
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.early, t));
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.early_whole, t));
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.auto, t));
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.invalid, t));
    try ss.installReadSchedule(.current, t);
    var opts = options(&rows, .host);
    try testing.expectError(error.MimoPrefillReadSchedule, applyReadSchedule(&opts, .invalid));
    try applyReadSchedule(&opts, .early);
    try testing.expectEqual(io_tickets_early, opts.pool.tickets);
    // The admissions keep the stock ring's bound.
    try testing.expectEqual(@as(?u32, (io_tickets - 2 * sdk_ext.expert.io.max_items) / 2), opts.ahead_cap);
}

test "mimo p7 read schedule: a stream built for early refuses early_whole and current; early_whole sets the whole read-ahead post" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var f: HitFixture = undefined;
    var ss = try p7Fixture(&f, a, &rows, .early);
    defer f.deinit(a);
    const t = default_read_schedule_threshold;
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.early_whole, t));
    try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.current, t));
    try ss.installReadSchedule(.early, t);
    var opts = options(&rows, .host);
    try applyReadSchedule(&opts, .early_whole);
    try testing.expectEqual(S.AheadPost.whole, opts.prompt_ahead);
    try testing.expectEqual(S.PromptReads.overlap, opts.prompt_reads);
}

// ── X20 (mimo/sched-auto) CPU proofs: `prefill_read_schedule: auto` on the same host stream fixture. ──

test "mimo p7 read schedule auto: early below the 8,192-token threshold, early_whole at or above (1,023 / 4,096 / 8,191 / 8,192 / 16,384), set on the stream once per request; a named schedule sets nothing" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    const lengths = [_]u64{ 1023, 4096, 8191, 8192, 16_384 };
    const want = [_]ReadSchedule{ .early, .early, .early, .early_whole, .early_whole };
    // The table alone, at the default threshold.
    const t: AutoSchedule = .{};
    try testing.expectEqual(@as(u32, 8192), t.threshold);
    for (lengths, want) |n, w| try testing.expectEqual(w, t.of(n));
    // `auto` builds `early`'s stream: the overlap schedule on the wider ring, the stock ring's admissions.
    var opts = options(&rows, .host);
    var early_opts = options(&rows, .host);
    try applyReadSchedule(&opts, .auto);
    try applyReadSchedule(&early_opts, .early);
    try testing.expectEqual(early_opts.prompt_reads, opts.prompt_reads);
    try testing.expectEqual(early_opts.prompt_ahead, opts.prompt_ahead);
    try testing.expectEqual(early_opts.pool.tickets, opts.pool.tickets);
    try testing.expectEqual(early_opts.ahead_cap, opts.ahead_cap);
    {
        // Installed: each request's start sets the stream's read-ahead post (head below, whole at or above), in any order.
        var f: HitFixture = undefined;
        var ss = try p7Fixture(&f, a, &rows, .auto);
        defer f.deinit(a);
        try testing.expect(ss.early);
        for (0..2) |_| for (lengths, want) |n, w| {
            try testing.expectEqual(w, try ss.chooseReadSchedule(n));
            try testing.expectEqual(aheadPost(w), f.stream.prompt_ahead);
        };
        // The threshold's bounds: 1 (every prompt early_whole) and 1,048,576 (early through 1,048,575); 0 and past the
        // bound refuse at installation.
        try ss.installReadSchedule(.auto, 1);
        try testing.expectEqual(ReadSchedule.early_whole, try ss.chooseReadSchedule(1));
        try ss.installReadSchedule(.auto, max_read_schedule_threshold);
        try testing.expectEqual(ReadSchedule.early, try ss.chooseReadSchedule(max_read_schedule_threshold - 1));
        try testing.expectEqual(ReadSchedule.early_whole, try ss.chooseReadSchedule(max_read_schedule_threshold));
        try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.auto, 0));
        try testing.expectError(error.MimoPrefillReadSchedule, ss.installReadSchedule(.auto, max_read_schedule_threshold + 1));
    }
    // A named schedule is every request's: a request's start returns it and sets nothing.
    for ([_]ReadSchedule{ .current, .early, .early_whole }) |named| {
        var f: HitFixture = undefined;
        var ss = try p7Fixture(&f, a, &rows, named);
        defer f.deinit(a);
        const post = f.stream.prompt_ahead;
        for (lengths) |n| {
            try testing.expectEqual(named, try ss.chooseReadSchedule(n));
            try testing.expectEqual(post, f.stream.prompt_ahead);
        }
    }
}

test "mimo p7 read schedule auto CPU: below the threshold every request runs exactly as early, at or above exactly as early_whole (plans, residency, hits, misses, bytes, drops); a mixed run's requests each run their own" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    const n_calls = P7Calls40.n_calls;
    var c40: P7Calls40 = .{};
    c40.fill(rows[0]);
    // 10 requests of 4 calls (the first proof's script); `auto` at 1,023 tokens, at 16,384, and cycling the five lengths
    // (longest first: the script's one dropped record falls in a request that takes `early`).
    const base: P7Script = .{ .ids = &c40.ids, .pred = &c40.pred, .per_prompt = 4 };
    var s_lo = base;
    s_lo.prompt_tokens = &.{1023};
    var s_hi = base;
    s_hi.prompt_tokens = &.{16_384};
    var s_mix = base;
    const lengths = [_]u64{ 16_384, 8192, 8191, 4096, 1023 };
    s_mix.prompt_tokens = &lengths;
    const calls = try a.alloc([n_calls]P7Call, 6);
    defer a.free(calls);
    var runs: [6]P7Run = undefined;
    for (&runs, calls) |*r, *c| r.* = .{ .calls = c, .stats = .{}, .residents = undefined };
    const cur = &runs[0];
    const early = &runs[1];
    const whole = &runs[2];
    const lo = &runs[3];
    const hi = &runs[4];
    const mix = &runs[5];
    try p7RunScript(a, .current, &rows, base, cur);
    try p7RunScript(a, .early, &rows, base, early);
    try p7RunScript(a, .early_whole, &rows, base, whole);
    try p7RunScript(a, .auto, &rows, s_lo, lo);
    try p7RunScript(a, .auto, &rows, s_hi, hi);
    try p7RunScript(a, .auto, &rows, s_mix, mix);
    // Every run plans every call as current does and leaves its residency, its hits and its misses.
    for (runs[1..]) |*run| {
        for (cur.calls, run.calls) |*want_call, *got| try p7ExpectSamePlans(want_call, got);
        for (cur.residents, run.residents) |rc, rr| try testing.expectEqualSlices(u16, &rc, &rr);
        try testing.expectEqual(cur.stats.expert_cache_hits, run.stats.expert_cache_hits);
        try testing.expectEqual(cur.stats.expert_cache_misses, run.stats.expert_cache_misses);
    }
    // Below the threshold: early's run exactly; at or above: early_whole's (bytes, drops, posts, the posts ahead of the
    // executor, each request's own bytes and drops).
    for ([_][2]*const P7Run{ .{ early, lo }, .{ whole, hi } }) |pair| {
        const ref = pair[0];
        const got = pair[1];
        try testing.expectEqual(ref.stats.expert_bytes_read, got.stats.expert_bytes_read);
        try testing.expectEqual(ref.stats.ahead_dropped, got.stats.ahead_dropped);
        try testing.expectEqual(ref.stats.ahead_posted, got.stats.ahead_posted);
        for (ref.calls, got.calls) |*x, *y| try testing.expectEqual(x.posted_ahead, y.posted_ahead);
        try testing.expectEqual(ref.n_requests, got.n_requests);
        try testing.expectEqualSlices(u64, ref.req_bytes[0..ref.n_requests], got.req_bytes[0..got.n_requests]);
        try testing.expectEqualSlices(u64, ref.req_dropped[0..ref.n_requests], got.req_dropped[0..got.n_requests]);
    }
    for (lo.chosen[0..lo.n_requests]) |c| try testing.expectEqual(ReadSchedule.early, c);
    for (hi.chosen[0..hi.n_requests]) |c| try testing.expectEqual(ReadSchedule.early_whole, c);
    // Mixed: request r chose by its own length and read exactly what that schedule reads for it (each request starts
    // from the same rows under every schedule: the plans are the same, and a dropped record's row takes the load that
    // evicted it, as under current).
    try testing.expectEqual(@as(usize, n_calls / 4), mix.n_requests);
    const want = [_]ReadSchedule{ .early_whole, .early_whole, .early, .early, .early };
    var mix_early: u32 = 0;
    for (0..mix.n_requests) |r| {
        try testing.expectEqual(want[r % want.len], mix.chosen[r]);
        const ref = if (mix.chosen[r] == .early) early else whole;
        try testing.expectEqual(ref.req_bytes[r], mix.req_bytes[r]);
        try testing.expectEqual(ref.req_dropped[r], mix.req_dropped[r]);
        mix_early += @intFromBool(mix.chosen[r] == .early);
    }
    // The script exercises the difference (early drops what early_whole reads twice, as current does).
    try testing.expectEqual(cur.stats.expert_bytes_read, whole.stats.expert_bytes_read);
    try testing.expect(early.stats.expert_bytes_read < whole.stats.expert_bytes_read);
    try testing.expect(mix.stats.expert_bytes_read >= early.stats.expert_bytes_read and mix.stats.expert_bytes_read <= whole.stats.expert_bytes_read);
    // ... and the mixed run holds both: early requests (with the drop) and early_whole requests.
    try testing.expect(mix_early > 0 and mix_early < mix.n_requests and mix.stats.ahead_dropped == early.stats.ahead_dropped);
    std.debug.print("MIMO_P7_AUTO calls={d} requests={d} threshold={d} bytes current={d} early={d} early_whole={d} auto_1023={d} auto_16384={d} auto_mixed={d} mixed_early_requests={d} dropped early={d} auto_1023={d} auto_16384={d} auto_mixed={d}\n", .{
        n_calls,                     mix.n_requests,                default_read_schedule_threshold,
        cur.stats.expert_bytes_read, early.stats.expert_bytes_read, whole.stats.expert_bytes_read,
        lo.stats.expert_bytes_read,  hi.stats.expert_bytes_read,    mix.stats.expert_bytes_read,
        mix_early,                   early.stats.ahead_dropped,     lo.stats.ahead_dropped,
        hi.stats.ahead_dropped,      mix.stats.ahead_dropped,
    });
}

/// X20: a choice past a request's start refuses (`Stream.setPromptAhead`) and leaves the request's post, either length.
fn p7NoChoice(f: *HitFixture, ss: *StreamSource) anyerror!void {
    const post = f.stream.prompt_ahead;
    for ([_]u64{ 1023, 16_384 }) |n| {
        if (ss.chooseReadSchedule(n)) |_| return error.P7TestChoiceMidRequest else |e| if (e != error.RoutesLive and e != error.PromptStarted) return e;
    }
    try std.testing.expectEqual(post, f.stream.prompt_ahead);
}

test "mimo p7 read schedule auto: a request's schedule never changes after its start: a choice mid-pass refuses (read-ahead in flight, call planned, between calls, after the pass) and leaves the post; the next request's start chooses again; a stock stream takes no post" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var c40: P7Calls40 = .{};
    c40.fill(rows[0]);
    {
        var f: HitFixture = undefined;
        var ss = try p7Fixture(&f, a, &rows, .auto);
        defer f.deinit(a);
        var out: P7Call = .{};
        for ([_]u64{ 1023, 16_384, 8191, 8192 }, 0..) |prompt, req| {
            const chose = try ss.chooseReadSchedule(prompt);
            try testing.expectEqual((AutoSchedule{}).of(prompt), chose);
            // The request's four calls, each probed mid-call (read-ahead in flight; call planned) and after it.
            for (0..4) |k| {
                const i = req * 4 + k;
                try p7RunCallProbed(&f, &ss, @intCast(1 + i % 2), c40.ids[i], c40.pred[i], &out, p7NoChoice);
                try p7NoChoice(&f, &ss);
                try testing.expectEqual(aheadPost(chose), f.stream.prompt_ahead);
            }
            // The pass settled (nothing live), its residents still held: still this request's.
            try ss.stream.settleRoutes();
            try testing.expectError(error.PromptStarted, ss.chooseReadSchedule(if (chose == .early) 16_384 else 1023));
            try testing.expectEqual(aheadPost(chose), f.stream.prompt_ahead);
            // The next request's start (`Source.beginPrefill`: residents forgotten, counts zeroed).
            _ = try ss.forgetResidents();
            ss.resetPromptCounts();
        }
    }
    // A stock stream (`current`; built after the auto fixture's pool stopped: one pool per process) refuses a post by
    // name; its requests run the stock schedule.
    var g: HitFixture = undefined;
    var cs = try p7Fixture(&g, a, &rows, .current);
    defer g.deinit(a);
    try testing.expectError(error.InvalidOptions, g.stream.setPromptAhead(.whole));
    try testing.expectEqual(ReadSchedule.current, try cs.chooseReadSchedule(16_384));
}

test "mimo p7 read schedule auto with kept residents: request 2 chooses its schedule over request 1's residents (prompt_residents: keep), and a seeded stream's request 1 chooses over the seed; a choice after the prompt started still refuses" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 28, 28 };
    var c40: P7Calls40 = .{};
    c40.fill(rows[0]);
    var f: HitFixture = undefined;
    var ss = try p7Fixture(&f, a, &rows, .auto);
    defer f.deinit(a);
    var out: P7Call = .{};
    for ([_]u64{ 1023, 16_384, 8191 }, 0..) |prompt, req| {
        // Request 1 starts after the construction's forget; each later one keeps the last one's residents
        // (`Source.beginPrefill` under keep: `Stream.forgetPrompt`, then the counts zeroed).
        if (req > 0) try testing.expect(f.stream.residentCount() > 0);
        const chose = try ss.chooseReadSchedule(prompt);
        try testing.expectEqual((AutoSchedule{}).of(prompt), chose);
        try testing.expectEqual(aheadPost(chose), f.stream.prompt_ahead);
        for (0..2) |k| try p7RunCallProbed(&f, &ss, @intCast(1 + k), c40.ids[req * 2 + k], c40.pred[req * 2 + k], &out, p7NoChoice);
        try ss.stream.settleRoutes();
        try testing.expectError(error.PromptStarted, ss.chooseReadSchedule(prompt));
        try f.stream.forgetPrompt();
        ss.resetPromptCounts();
    }
    // A seeded stream (residents read at the load, no prompt yet): request 1 chooses.
    _ = try ss.forgetResidents();
    const Order = struct {
        ids: []const u16,
        fn of(o: @This(), _: u32) []const u16 {
            return o.ids;
        }
    };
    _ = try f.stream.seedResidents(Order{ .ids = &.{ 0, 1, 2, 3 } }, Order.of);
    ss.resetPromptCounts();
    try testing.expect(f.stream.residentCount() > 0);
    try testing.expectEqual((AutoSchedule{}).of(16_384), try ss.chooseReadSchedule(16_384));
}

fn packedRef(ref: SlotRef) u32 {
    return (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
}

/// One call's route, its parts landed, released and flushed (the scheduler's sequence, without math).
fn hitRoute(ss: *StreamSource, layer: u32, ids: []const u16) !void {
    const call = try ss.route(layer, ids, &.{});
    for (0..call.route.?.n_parts) |p| {
        try ss.waitGu(call, @intCast(p));
        try ss.waitDown(call, @intCast(p));
    }
    ss.release(call);
    try ss.flush();
}

/// The LUT as the policy defines it: each resident expert's persistent slot (packed), else 0.
fn expectLutIsPolicy(ss: *const StreamSource, layer: u32, lut: []const u32) !void {
    const rel = ss.relativeLayer(layer);
    const pol = &ss.stream.layers[rel].policy;
    for (lut, 0..) |v, e| {
        const want: u32 = if (pol.slotOf(@intCast(e))) |s| (if (s < pol.capacity) packedRef(ss.stream.slotRef(rel, s)) else 0) else 0;
        try std.testing.expectEqual(want, v);
    }
}

test "mimo hitpass: residentLut stays the policy's map under a scripted admit/evict sequence (dense layer 0 all misses)" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 4, 4 };
    const f = try a.create(HitFixture);
    defer a.destroy(f);
    try f.init(a, &rows);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    try ss.grow(&rows);
    try testing.expect(ss.decoding());
    var lut: [3][64]u32 = undefined;
    for (0..3) |l| ss.residentLut(@intCast(l), &lut[l]);
    // The dense layer has no policy: every expert reads base row 0 and is never joined.
    try testing.expectEqualSlices(u32, &@as([64]u32, @splat(0)), &lut[0]);
    // A scripted sequence over 4-row layers: repeats (hits), new experts (admits), and evictions of the oldest.
    const script = [_][8]u16{
        .{ 0, 1, 2, 3, 4, 5, 6, 7 },     .{ 0, 1, 2, 3, 8, 9, 10, 11 },     .{ 8, 9, 10, 11, 0, 1, 12, 13 },
        .{ 12, 13, 8, 9, 0, 1, 2, 3 },   .{ 40, 41, 42, 43, 8, 9, 10, 11 }, .{ 40, 41, 0, 1, 2, 3, 63, 62 },
        .{ 63, 62, 40, 41, 5, 6, 7, 4 }, .{ 4, 5, 6, 7, 63, 62, 61, 60 },
    };
    var changed: u32 = 0;
    for (0..2) |round| for (script, 0..) |ids_in, i| {
        const layer: u32 = @intCast(1 + (i + round) % 2);
        var ids = ids_in;
        if (round == 1) for (&ids) |*e| {
            e.* = 63 - e.*;
        };
        const before = lut[layer];
        try expectLutIsPolicy(&ss, layer, &before);
        try hitRoute(&ss, layer, &ids);
        ss.residentLut(layer, &lut[layer]);
        try expectLutIsPolicy(&ss, layer, &lut[layer]);
        changed += @intFromBool(!std.mem.eql(u32, &before, &lut[layer]));
    };
    // The policy moved (admits and evictions), and the LUT moved with it on every route that changed residency.
    try testing.expect(changed >= 8);
    // The dense layer is untouched by routed traffic.
    ss.residentLut(0, &lut[0]);
    try testing.expectEqualSlices(u32, &@as([64]u32, @splat(0)), &lut[0]);
}

test "mimo hitpass: the hit/miss split on synthetic ids: every hit's LUT entry (written after the layer's previous route) is the (bank, row) the host serves it from; every miss is non-resident" {
    const testing = std.testing;
    const a = testing.allocator;
    const rows = [_]u32{ 4, 4 };
    const f = try a.create(HitFixture);
    defer a.destroy(f);
    try f.init(a, &rows);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    try ss.grow(&rows);
    var lut: [3][64]u32 = undefined;
    for (0..3) |l| ss.residentLut(@intCast(l), &lut[l]);
    var rng = std.Random.DefaultPrng.init(20261008);
    var n_hits: usize = 0;
    var n_misses: usize = 0;
    var wide_stale: usize = 0;
    for (0..300) |call_i| {
        const layer: u32 = @intCast(1 + call_i % 2);
        // Every 7th call is a decode-phase wide route (the MTP verify rows: 2 tokens x top-8): it moves the policy
        // outside the one-token lane, which is why `Source.lutAfterWide` rewrites the LUT after it.
        const n_ids: usize = if (call_i % 7 == 6) 16 else 8;
        var ids: [16]u16 = undefined;
        for (ids[0..n_ids], 0..) |*d, j| {
            // Skewed: a hot set of 6 (repeats: hits) and a cold tail (admits, evictions); distinct per token.
            while (true) {
                const e: u16 = if (rng.random().uintLessThan(u8, 3) == 0) rng.random().uintLessThan(u16, 64) else rng.random().uintLessThan(u16, 6);
                const tok = j / 8;
                if (std.mem.indexOfScalar(u16, ids[tok * 8 .. j], e) == null) {
                    d.* = e;
                    break;
                }
            }
        }
        var resident_before: [64]bool = undefined;
        for (&resident_before, 0..) |*r, e| r.* = ss.isResident(layer, @intCast(e));
        const call = try ss.route(layer, ids[0..n_ids], &.{});
        const sv = ss.served(call);
        for (ids[0..n_ids], sv.refs, sv.waves) |e, ref, w| {
            if (w == 0) {
                // A hit: the device wave reads exactly the slot the host's plan serves.
                try testing.expect(resident_before[e]);
                try testing.expectEqual(packedRef(ref), lut[layer][e]);
                n_hits += 1;
            } else {
                // A miss: not resident when the LUT was written (its device row is replaced by its part's).
                try testing.expect(!resident_before[e]);
                n_misses += 1;
            }
        }
        for (0..call.route.?.n_parts) |p| {
            try ss.waitGu(call, @intCast(p));
            try ss.waitDown(call, @intCast(p));
        }
        ss.release(call);
        try ss.flush();
        var after: [64]u32 = undefined;
        ss.residentLut(layer, &after);
        if (n_ids == 16) wide_stale += @intFromBool(!std.mem.eql(u32, &after, &lut[layer]));
        lut[layer] = after;
    }
    try testing.expect(n_hits > 100 and n_misses > 100);
    // Without the rewrite, a wide route would leave a stale LUT for the next one-token call.
    try testing.expect(wide_stale > 0);
}

/// The scheduler's math on the trace backend: MiMo's projection type, the banked waves' shapes only.
const HitTraceMath = struct {
    pub const has_banked = true;
    pub const decode_forward_rows: u32 = 1;
    pub fn Projection(comptime A: type) type {
        return sdk_ext.quant.GatherQmm.Arrays(A);
    }
    hidden: c_int,
    inter: c_int,
    const Tr = @import("../ops.zig").TraceOps;
    pub fn gateUp(self: *const HitTraceMath, g: *Tr, x: u32, _: u32, _: Projection(u32), _: Projection(u32)) !u32 {
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
    pub fn down(self: *const HitTraceMath, g: *Tr, h: u32, _: u32, _: Projection(u32)) !u32 {
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .bfloat16);
    }
    pub fn gateUpBanked(self: *const HitTraceMath, g: *Tr, x: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
    pub fn downBanked(self: *const HitTraceMath, g: *Tr, h: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .bfloat16);
    }
};

const HitScript = struct {
    calls: []const []const u16,
    next: usize = 0,
    fn values(self: *HitScript) HitTraceMath.Tr.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *HitScript = @ptrCast(@alignCast(ctx));
        if (self.next >= self.calls.len) return error.ScriptExhausted;
        const c = self.calls[self.next];
        self.next += 1;
        if (c.len != out.len) return error.ScriptShape;
        @memcpy(out, c);
    }
    fn argmax(_: *anyopaque) anyerror!u32 {
        return error.NoPicks;
    }
};


// ── end HITPASS CPU tests ──

// The host's call sequence for an MTP-lane request against the stream (window lev1, cell mtp-typ-b1: `decode tick
// failed: NotPrefill`): the prompt pass, the phase change, the first decode token, then the round's verify, a
// decode-phase wide call (logical M = 4: 32 routed ids), made the way `ExpertsWith.runWideCore` makes it (the
// read-ahead's await, the seed, the hot order, the route). Before the fix the seed refused in decode; the prompt's
// read-ahead still does, so the module's verify routes none (`module.verify_read_ahead`). Host memory, CPU only.
test "mimo mtp lane: prefill -> first decode tick -> the verify's decode-phase wide call routes without a prompt seed (host)" {
    const testing = std.testing;
    const a = testing.allocator;
    const geom = try bank_mod.Geometry.init(32, 32, 64, 1, 2);
    const bytes = try a.alloc(u8, geom.total_bytes);
    defer a.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i * 7);
    var side: bank_mod.TestSidecar = undefined;
    try side.init(a, geom, bytes);
    defer side.deinit();
    {
        const rows = [_]u32{ 4, 4 };
        const stream = try Stream.init(a, &side.bank, options(&rows, .host));
        defer stream.deinit();
        var source = StreamSource.init(stream);
        const Wide = struct {
            /// One wide call as `runWideCore` makes it: await, seed, hot order, route, drain.
            fn call(src: *StreamSource, layer: u32, ids: []const u16) !void {
                try src.awaitReadAhead(layer);
                try src.seedPrefill(layer, ids);
                var count: [64]u32 = @splat(0);
                var distinct: [64]u16 = undefined;
                var n: usize = 0;
                for (ids) |e| {
                    if (count[e] == 0) {
                        distinct[n] = e;
                        n += 1;
                    }
                    count[e] += 1;
                }
                src.hotOrder(layer, distinct[0..n], &count);
                const c = try src.route(layer, distinct[0..n], &.{});
                for (0..src.served(c).n_parts) |p| {
                    try src.waitGu(c, @intCast(p));
                    try src.waitDown(c, @intCast(p));
                }
                src.release(c);
                try src.flush();
            }
        };
        // The prompt pass (prefill phase).
        for ([_]u32{ 1, 2 }) |layer| try Wide.call(&source, layer, &seed_prompt);
        // The phase change (`Source.beginDecode`).
        try source.grow(&rows);
        try testing.expect(source.decoding());
        // The first decode tick: one token, top-8.
        for ([_]u32{ 1, 2 }) |layer| {
            const one = [_]u16{ 4, 5, 0, 1, 20, 21, 22, 23 };
            const c = try source.route(layer, &one, &.{});
            for (0..source.served(c).n_parts) |p| {
                try source.waitGu(c, @intCast(p));
                try source.waitDown(c, @intCast(p));
            }
            source.release(c);
            try source.flush();
        }
        // The round's verify: 4 rows x top-8 in the decode phase. The prompt's read-ahead is refused here (the module
        // routes none on the verify); the wide call's seed is a decode no-op and its order the stock count order.
        try testing.expectError(error.NotPrefill, source.readAheadSeed(1, &.{ 30, 31 }));
        var verify: [4 * 8]u16 = undefined;
        for (&verify, 0..) |*e, i| e.* = @intCast((i * 5 + 3) % 40);
        for ([_]u32{ 1, 2 }) |layer| try Wide.call(&source, layer, &verify);
        try testing.expect(source.decoding());
        // The next decode tick still routes.
        const c = try source.route(1, &[_]u16{ 4, 5, 0, 1, 6, 7, 8, 9 }, &.{});
        for (0..source.served(c).n_parts) |p| {
            try source.waitGu(c, @intCast(p));
            try source.waitDown(c, @intCast(p));
        }
        source.release(c);
        try source.flush();
    }
}

/// Lever 2's synthetic prompt (8 tokens x top-8 over 64 experts): six tokens route {0..7}, the last two {8..11, 4, 5,
/// 12, 13}. Whole counts: 4, 5 -> 8; 0-3, 6, 7 -> 6; 8-13 -> 2. Last two tokens: 4, 5, 8-13 -> 2.
const seed_prompt = blk: {
    var ids: [8 * 8]u16 = undefined;
    for (0..6) |t| for (0..8) |j| {
        ids[t * 8 + j] = j;
    };
    for (6..8) |t| for ([_]u16{ 8, 9, 10, 11, 4, 5, 12, 13 }, 0..) |e, j| {
        ids[t * 8 + j] = e;
    };
    break :blk ids;
};

// Host-memory stream (two routed layers, 64 experts, 4 rows): the prompt pass routes each layer's experts in the
// installed hot order, so decode starts with exactly the installed seed. CPU only: host slot rows and pread workers.
