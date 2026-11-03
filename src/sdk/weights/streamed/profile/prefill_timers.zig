//! The DSV4.1 prompt pass's routed-call split, for PROFILE builds only (`-Ddsv41-prefill-timers=true`): host
//! time in the wide lane's steps and the DIG-X dispatcher's waves, accumulated per process. In every
//! other build `enabled` is false and each call below compiles to nothing (the timed path carries no
//! timer, branch or counter).

const std = @import("std");
const bo = @import("../build_flags.zig");
const sdk_ext = @import("../sdk_ext.zig");

pub const enabled: bool = if (@hasDecl(bo, "dsv41_prefill_timers")) bo.dsv41_prefill_timers else false;

/// Where the routed call's host time goes (the SDK's probe shape, G7): the routing barrier, the stream's route,
/// the read waits, the waves' encode, the drains, the join.
pub const Bucket = sdk_ext.profile.PrefillBucket;
const n_buckets = @typeInfo(Bucket).@"enum".field_names.len;

pub var ns: [n_buckets]u64 = @splat(0);
pub var waves: u64 = 0;
pub var calls: u64 = 0;
pub var launches: u64 = 0;

/// Whose time a charge is (PREFILL_PROFILE_ROUTED_SPLIT): the stream groups', or the deferred base call's at the seed
/// (P1b) or at the call's end. The wide lane sets it around the base call; the dispatcher's charges follow it.
pub const Mode = enum { stream, base_seed, base_end };
const n_modes = @typeInfo(Mode).@"enum".field_names.len;
pub var mode: Mode = .stream;
pub var ns_mode: [n_modes][n_buckets]u64 = @splat(@splat(0));
pub var mode_calls: [n_modes]u64 = @splat(0);
pub var wide_calls: u64 = 0;

/// P1's read-ahead, one record per layer (PREFILL_PROFILE_READAHEAD): the predictor's counts, the predicted seed
/// (the top of its ranking the layer's empty rows hold), the admissions and reads, the predicted experts no empty row
/// took (blocked: every row held by a resident), and at the barrier the hits, the seed's demand records and their
/// predicted counts against the cut (the last predicted rank's count): within x0.8-1.25 (near) or beyond (far).
pub const max_layers = @import("decode_timers.zig").max_layers;
pub const max_experts = 512;
pub const ReadAheadLayer = struct { predicted: u32 = 0, admitted: u32 = 0, posted: u32 = 0, blocked: u32 = 0, cut: u32 = 0, hits: u32 = 0, demand: u32 = 0, near: u32 = 0, far: u32 = 0 };
pub var ra: [max_layers]ReadAheadLayer = @splat(.{});
pub var ra_counts: [max_layers][max_experts]u32 = @splat(@splat(0));

const clock = sdk_ext.profile.Clock(enabled);
pub const Stamp = clock.Stamp;
pub const now = clock.now;

/// Charge the time since `t0` to `b` (and to the current mode's split).
pub inline fn charge(b: Bucket, t0: Stamp) void {
    if (comptime !enabled) return;
    const d = now() - t0;
    ns[@intFromEnum(b)] += d;
    ns_mode[@intFromEnum(mode)][@intFromEnum(b)] += d;
}

/// The charges that follow are `m`'s (the wide lane's base call sets it and restores `.stream`).
pub inline fn setMode(m: Mode) void {
    if (comptime !enabled) return;
    mode = m;
    if (m != .stream) mode_calls[@intFromEnum(m)] += 1;
}

pub inline fn countWideCall() void {
    if (comptime !enabled) return;
    wide_calls += 1;
}

pub fn modeSeconds(m: Mode, b: Bucket) f64 {
    return @as(f64, @floatFromInt(ns_mode[@intFromEnum(m)][@intFromEnum(b)])) / 1e9;
}

/// The predictor's counts for `layer` (before its read-ahead is posted).
pub inline fn recordPrediction(layer: usize, counts: []const u32) void {
    if (comptime !enabled) return;
    if (layer >= max_layers) return;
    const n = @min(counts.len, max_experts);
    @memcpy(ra_counts[layer][0..n], counts[0..n]);
}

pub inline fn recordAdmission(layer: usize, predicted: u32, admitted: u32, blocked: u32, cut: u32) void {
    if (comptime !enabled) return;
    if (layer >= max_layers) return;
    const r = &ra[layer];
    r.predicted = predicted;
    r.admitted = admitted;
    r.blocked = blocked;
    r.cut = cut;
    r.posted = 0;
}

pub inline fn addPosted(layer: usize, n: u32) void {
    if (comptime !enabled) return;
    if (layer >= max_layers) return;
    ra[layer].posted += n;
}

pub inline fn recordBarrier(layer: usize, hits: u32, demand: u32, near: u32, far: u32) void {
    if (comptime !enabled) return;
    if (layer >= max_layers) return;
    const r = &ra[layer];
    r.hits = hits;
    r.demand = demand;
    r.near = near;
    r.far = far;
}

/// The stream's read-ahead probes (`expert_stream.ReadAheadProbe`, G7): the source hands over what it saw; the
/// predictor's counts, the cut and the near / far bands are this file's.
pub fn readAheadBarrier(layer: u32, hits: u32, seed: *const std.DynamicBitSetUnmanaged) void {
    if (comptime !enabled) return;
    var near: u32 = 0;
    var far: u32 = 0;
    if (layer < max_layers) {
        const cut = ra[layer].cut;
        var it = seed.iterator(.{});
        while (it.next()) |e| {
            const pc = if (e < max_experts) ra_counts[layer][e] else 0;
            if (nearCut(pc, cut)) near += 1 else far += 1;
        }
    }
    recordBarrier(layer, hits, @intCast(seed.count()), near, far);
}

pub fn readAheadAdmission(layer: u32, top: []const u16, admitted: u32, blocked: u32) void {
    if (comptime !enabled) return;
    const cut: u32 = if (top.len > 0 and layer < max_layers and top[top.len - 1] < max_experts) ra_counts[layer][top[top.len - 1]] else 0;
    recordAdmission(layer, @intCast(top.len), admitted, blocked, cut);
}

pub fn readAheadPosted(layer: u32, n: u32) void {
    addPosted(layer, n);
}

/// A demand record's band: its predicted count within x0.8-1.25 of the cut (near), else far.
pub fn nearCut(predicted: u32, cut: u32) bool {
    return cut > 0 and @as(u64, predicted) * 5 >= @as(u64, cut) * 4 and @as(u64, predicted) * 4 <= @as(u64, cut) * 5;
}

pub inline fn count(w: u64, l: u64) void {
    if (comptime !enabled) return;
    waves += w;
    launches += l;
}

pub inline fn countCall() void {
    if (comptime !enabled) return;
    calls += 1;
}

pub fn reset() void {
    ns = @splat(0);
    waves = 0;
    calls = 0;
    launches = 0;
    mode = .stream;
    ns_mode = @splat(@splat(0));
    mode_calls = @splat(0);
    wide_calls = 0;
    ra = @splat(.{});
}

pub fn seconds(b: Bucket) f64 {
    return @as(f64, @floatFromInt(ns[@intFromEnum(b)])) / 1e9;
}

test "dsv41 prefill timers: a demand record's band against the read-ahead's cut (x0.8-1.25 near, beyond far)" {
    const t = @import("std").testing;
    try t.expect(nearCut(100, 100) and nearCut(80, 100) and nearCut(125, 100));
    try t.expect(!nearCut(79, 100) and !nearCut(126, 100) and !nearCut(0, 100));
    // No cut (nothing predicted): every demand record is far.
    try t.expect(!nearCut(5, 0));
    // Every call compiles to nothing in a timed build: the records stay empty.
    if (!enabled) {
        recordAdmission(0, 5, 4, 1, 10);
        recordBarrier(0, 3, 2, 1, 1);
        try t.expectEqual(@as(u32, 0), ra[0].predicted);
    }
}
