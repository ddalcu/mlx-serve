//! G6, the expert_source kind's shared surface: the record layout a bank fills when it opens, the contract every
//! source satisfies (checked at comptime, the proposal's negotiation step 3), and the slot, gate and counter types an
//! arch's MoE math binds. A source fetches; the MoE math is its arch's. Every call a source takes runs on the
//! inference thread; nothing here is asked per token or per layer at run time.

const std = @import("std");
const mlx = @import("sdk").mlx;

/// The reader: one pool per process (lib/expert_io's C pool), read through one record topology per source
/// (`Records(components, gate_up)`), from fds `openUncached` makes. On graphs without the C sources it refuses at start.
pub const io = @import("expert/io.zig");
pub const Pool = io.Pool;
pub const Records = io.Records;
pub const UncachedFd = io.UncachedFd;
pub const openUncached = io.openUncached;
pub const openUncachedFollowing = io.openUncachedFollowing;
pub const max_range_components = io.max_range_components;
pub const checkTopology = io.checkTopology;

/// The event gate: the reader signals it, the arch's math waits on it (an MTLSharedEvent, or a host word on CPU
/// streams).
pub const event = @import("expert/event.zig");
pub const Event = event.Event;

/// The residency policy every source plans with (one admission and replacement policy for every source).
pub const policy = @import("expert/policy.zig");
/// The expert stream over a bank module (`assertBank`): slot rows, routes, residency, reads, lookahead, gates, release.
pub const stream = @import("expert/stream.zig");
/// A bank's open around its own manifest rules: the manifest read, the sidecar opened for the reader, the peek.
pub const bank_file = @import("expert/bank_file.zig");
/// The lookahead selector: the next routed layer's predicted records, chosen by score over its router rows.
pub const lookahead = @import("expert/lookahead.zig");
/// The route recorder (a diagnostic): one request's decode routes, what served each id, and the handover's residents.
pub const routes = @import("expert/routes.zig");
/// The rate tally's settings and file (every model's residency: the top share of each layer's experts by request rate
/// pinned, the rates carried from run to run under ~/.mlx-stream/<model key>/).
pub const tally = @import("expert/tally.zig");
/// The expert usage profile (`expert_usage_record`) and the resident seed (`resident_seed`) from the pack's
/// `expert-usage-v1.json`; `prompt_residents`, the reverse change's route.
pub const usage = @import("expert/usage.zig");

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ExpertState = struct {
    /// The one reader per process: the host takes it at the load claim of an arch whose caps say `uses_expert_reader`
    /// (its source declares `uses_reader`), before the preflight and the weights, and gives it back when the loaded model
    /// goes or the load fails. A second is refused by name, never at the pool's start.
    reader_taken: std.atomic.Value(bool) = .init(false),
};
pub const expertState_shared = @import("mlx_host").shared.Shared(ExpertState, "streamed_expert", .{});
inline fn expertState() *ExpertState {
    return expertState_shared.get();
}

pub fn takeReader() error{ExpertReaderInUse}!void {
    if (expertState().reader_taken.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return error.ExpertReaderInUse;
}

pub fn giveReader() void {
    expertState().reader_taken.store(false, .release);
}

/// The record layout a bank fills when it opens (EXL3: 9 components, 6 gate/up; MXFP4: 6, 4): each component's dtype,
/// per-row shape, offset in the record and length; the record's bytes and its gate/up range's.
pub const RecordLayout = struct {
    pub const Segment = struct { dtype: mlx.mlx_dtype, shape: [3]u64 = @splat(0), rank: u8, offset: u64, length: u64 };

    components: u8,
    gate_up: u8,
    segments: [2 * max_range_components]Segment,
    record_bytes: u64,
    gate_up_bytes: u64,

    pub fn segmentsOf(l: *const RecordLayout) []const Segment {
        return l.segments[0..l.components];
    }
};

/// Which of a layer's banks holds a slot: its prefill rows, the rows `grow` added, or the transient scratch every
/// layer shares (a source with one bank serves every slot from `.base`).
pub const BankKind = enum(u8) { base, ext, transient };
/// A slot's bank and its row in that bank (the index the kernels gather).
pub const SlotRef = struct { bank: BankKind, row: u32 };

/// What one routed-layer call serves, valid until the call is released. Per routed id, in the router's flat order:
/// the slot's bank and row, and the wave that computes it (0 = resident at the call; p + 1 = miss part p, whose
/// gate/up may run after `waitGu(p)` and its down after `waitDown(p)`).
pub const Served = struct {
    refs: []const SlotRef,
    waves: []const u8,
    n_parts: u32,
};

/// The event values a gated call's waves wait for: its gate/up wave `gu`, the down wave of part p `down_first + p`.
pub const Gates = struct { gu: u64, down_first: u64, n_parts: u32 };

/// A source's counters (the receipts read these names); a source reports zeros for the classes it lacks.
pub const Stats = struct {
    route_calls: u64 = 0,
    /// Unique experts per route that were resident / had to be loaded.
    expert_cache_hits: u64 = 0,
    expert_cache_misses: u64 = 0,
    expert_cache_evictions: u64 = 0,
    persistent_loads: u64 = 0,
    transient_loads: u64 = 0,
    /// Loads whose slot still held the record: no read.
    loads_skipped: u64 = 0,
    expert_bytes_read: u64 = 0,
    preadv_calls: u64 = 0,
    /// Sum over read ranges of first syscall to publication.
    expert_read_seconds: f64 = 0,
    /// The part of it spent by ranges served without a preadv (adopted out of a speculative record): copy only.
    adopt_copy_seconds: f64 = 0,
    /// Wall time with a read in flight (the pool's gauge).
    read_wall_ns: u64 = 0,
    /// Host time blocked in the read waits (a part's gate/up or its settle, a read-ahead's landing).
    read_wait_ns: u64 = 0,
    /// The lookahead class: records claimed by a demand read, physical bytes
    /// of speculative reads, records issued / fully landed, demand ranges
    /// copied out of a speculative record (no preadv) and their bytes.
    claimed: u64 = 0,
    spec_bytes: u64 = 0,
    spec_issued: u64 = 0,
    spec_landed: u64 = 0,
    /// The lookahead's records no demand read claimed: dropped while queued, abandoned while read, read and never
    /// served (the waste's attribution); demand ranges read straight into their rows (`io.Options.direct`).
    spec_expired: u64 = 0,
    spec_abandoned: u64 = 0,
    spec_discarded: u64 = 0,
    spec_cancelled: u64 = 0,
    /// Of the claimed, those still in flight at the claim (the prediction right, the read late); the physical bytes
    /// the abandoned records had read.
    spec_claimed_inflight: u64 = 0,
    spec_abandoned_bytes: u64 = 0,
    direct_ranges: u64 = 0,
    adopt_ranges: u64 = 0,
    adopt_bytes: u64 = 0,
    /// Pre-read ranges queued, served to a demand read, dropped unbound.
    pre_issued: u64 = 0,
    pre_served: u64 = 0,
    pre_expired: u64 = 0,
    /// Event gates registered and forced by the watchdog.
    gates: u64 = 0,
    gates_forced: u64 = 0,
    /// P1's read-ahead: records posted; at each barrier, the records read ahead that the call routes (its hits)
    /// and its seed's records not read ahead (its demand loads); the bytes read ahead (in `expert_bytes_read` too).
    ahead_posted: u64 = 0,
    ahead_hits: u64 = 0,
    ahead_demand: u64 = 0,
    ahead_bytes: u64 = 0,
    /// P7 overlap: read-ahead records never read because the call's plans evict them before any hit (`p7Tail`).
    ahead_dropped: u64 = 0,
    /// Prompt records copied from a transient row into a resident's row (`Stream.promote`) and their bytes.
    promoted: u64 = 0,
    promoted_bytes: u64 = 0,

    /// The counters since `prev`, an earlier snapshot of the same stream: each field's difference, at least zero.
    pub fn since(s: Stats, prev: Stats) Stats {
        var d: Stats = .{};
        inline for (@typeInfo(Stats).@"struct".field_names) |name| {
            const now = @field(s, name);
            const then = @field(prev, name);
            @field(d, name) = switch (@TypeOf(now)) {
                u64 => now -| then,
                f64 => @max(now - then, 0),
                else => @compileError("Stats." ++ name ++ ": a counter is u64 or f64"),
            };
        }
        return d;
    }
};

/// One layer's routes so far, every phase: the routes, the unique experts per route that were resident / had to be
/// loaded.
pub const LayerCounts = struct { routes: u64 = 0, hits: u64 = 0, misses: u64 = 0 };

/// A source's refusals at run time. A failure is sticky: every later call refuses as `StreamFailed`.
pub const Error = error{
    StreamFailed,
    RoutesLive,
    RoutesExhausted,
    SlotStillPinned,
    ReadFailed,
    Timeout,
    TicketsBusy,
    QueueFull,
    SubmitRefused,
    InvalidJob,
    SpecRefused,
    PreReadRefused,
    GateInvalid,
    GatesFull,
    GateRefused,
    /// The watchdog released a gate before its bytes landed: the GPU may have
    /// read them early, so the outputs since are invalid.
    GateForced,
    /// A decode route over more ids than decode's transient window holds rows.
    RouteWiderThanWindow,
};

/// What a source type supports (`S.caps`, comptime). An instance installs a subset at construction (a route chosen
/// per arm) and reports it once; no call asks again.
pub const Caps = struct {
    /// grow: the prompt rows become the decode rows once
    two_phase: bool = false,
    /// releaseTransient at the phase change
    transient_release: bool = false,
    /// a whole prompt's ids per layer before its routes (seedPrefill, seedRanks)
    prompt_seed: bool = false,
    /// the prompt pass reads a predicted seed ahead (readAheadSeed, awaitReadAhead)
    read_ahead: bool = false,
    /// several live calls per layer, each with its own window (holdBase, releaseHeld; wideDepth bounds them when
    /// declared)
    wide: bool = false,
    /// route's scores feed the selector and the speculative records (decode)
    lookahead: bool = false,
    /// a decode call pre-reads its certain misses
    preread: bool = false,
    /// the GPU waits on the reads' events instead of the host (gate)
    event_gates: bool = false,
    /// forgetResidents after the arch's install warm-up
    construction_reset: bool = false,
};

/// What source type `S` declares it supports (`S.caps`; none when undeclared).
pub fn capsOf(comptime S: type) Caps {
    return if (@hasDecl(S, "caps")) S.caps else .{};
}

/// Comptime: `S` is an expert source. A compile error names the first missing or mistyped declaration.
///   Required: `Call` (a live route); `route(*S, layer, ids, scores) Error!*Call` (scores: the next routed layer's
///   gate scores for a lookahead, or empty); `served(*S, *const Call) Served`; `waitGu` / `waitDown(*S, *Call, part)
///   Error!void`; `release(*S, *Call) void` (after the call's math is built); `flush(*S) Error!void` (after the
///   forward's last eval: the reuse fence); `stats(*S) Stats`; `bankRows(*S, layer, BankKind) u32`; `bankArrays`
///   (what the math binds, generic over the backend).
///   Per declared capability (`S.caps`): its methods, at today's signatures.
pub fn assertSource(comptime S: type) void {
    comptime {
        if (!@hasDecl(S, "Call")) @compileError(@typeName(S) ++ " is not an expert source: no Call");
        expectMethod(S, "route", &.{ *S, u32, []const u16, []const f32 }, *S.Call);
        expectMethod(S, "served", &.{ *S, *const S.Call }, Served);
        expectMethod(S, "waitGu", &.{ *S, *S.Call, u32 }, void);
        expectMethod(S, "waitDown", &.{ *S, *S.Call, u32 }, void);
        expectMethod(S, "release", &.{ *S, *S.Call }, void);
        expectMethod(S, "flush", &.{*S}, void);
        expectMethod(S, "stats", &.{*S}, Stats);
        expectMethod(S, "bankRows", &.{ *S, u32, BankKind }, u32);
        if (!@hasDecl(S, "bankArrays")) @compileError(@typeName(S) ++ " is not an expert source: no bankArrays");
        const caps = capsOf(S);
        if (caps.two_phase) expectMethod(S, "grow", &.{ *S, []const u32 }, void);
        if (caps.transient_release) expectMethod(S, "releaseTransient", &.{*S}, u64);
        if (caps.prompt_seed) {
            expectMethod(S, "seedPrefill", &.{ *S, u32, []const u16 }, void);
            expectMethod(S, "seedRanks", &.{ *const S, u32 }, u32);
        }
        if (caps.read_ahead) {
            expectMethod(S, "readAheadSeed", &.{ *S, u32, []const u16 }, void);
            expectMethod(S, "awaitReadAhead", &.{ *S, u32 }, void);
        }
        if (caps.wide) {
            expectMethod(S, "holdBase", &.{ *S, *S.Call }, void);
            expectMethod(S, "releaseHeld", &.{*S}, void);
        }
        if (@hasDecl(S, "wideDepth")) expectMethod(S, "wideDepth", &.{*const S}, u8);
        if (caps.event_gates) expectMethod(S, "gate", &.{ *S, *S.Call }, ?Gates);
        if (caps.construction_reset) expectMethod(S, "forgetResidents", &.{*S}, u32);
    }
}

/// The bank contract a stream reads (a bank MODULE `B`: one record file, fixed per-layer geometry):
/// - topology: `n_components`, `gu_components` (the gate/up range is segments [0, gu_components), the down range the rest,
///   each contiguous in the record), `Records` = `io.Records(n_components, gu_components)`, `Component` (an enum over the
///   segments);
/// - `Layer`: `segments: [n_components]` (each `offset` from the record start, `length`, `dtype`, `shape`, `rank`) and
///   `logical_bytes` (a record's bytes); `mlxDtype(dtype) mlx_dtype` (a slot array's dtype);
/// - `Bank`: fields `layers: []Layer` (the arch's routed layers), `n_experts`, `sidecar: UncachedFd`; methods
///   `recordOffset(*const Bank, layer, expert) u64` and `spans(*const Bank, layer, expert)` (`.gu_offset`, `.down_offset`);
/// - `BankArrays` and `bankArraysOf([n_components]mlx_array) BankArrays`: the slot arrays as the bank's quant binds them;
/// - `routed_top_k`: the arch's routed experts per token (the lookahead selector's per-row threshold rank);
/// - optional `scatter = true` (an MLX pack, whose record parts live apart): `Bank.parts(*const Bank, layer, expert)`
///   gives each component's file and offset (`Records.Part`), and `recordOffset` / `spans` only order the reads.
/// The stream allocates the slot rows, plans routes, reads, gates and releases; the arch keeps the MoE math.
pub fn assertBank(comptime B: type) void {
    comptime {
        const where = @typeName(B) ++ " is not an expert bank: ";
        for ([_][]const u8{ "n_components", "gu_components", "Records", "Component", "Layer", "Bank", "mlxDtype", "BankArrays", "bankArraysOf", "routed_top_k" }) |d|
            if (!@hasDecl(B, d)) @compileError(where ++ "no " ++ d);
        if (B.Records != io.Records(B.n_components, B.gu_components)) @compileError(where ++ "Records is not io.Records(n_components, gu_components)");
        if (!@hasField(B.Layer, "segments") or !@hasField(B.Layer, "logical_bytes")) @compileError(where ++ "Layer needs segments and logical_bytes");
        // A slot array is [rows] ++ a segment's shape: at most 3 axes of its own (the stream's shapes are [4]c_int).
        const Seg = @typeInfo(@FieldType(B.Layer, "segments")).array.child;
        if (@typeInfo(@FieldType(Seg, "shape")).array.len > 3) @compileError(where ++ "a segment shape has more than 3 axes");
        for ([_][]const u8{ "layers", "n_experts", "sidecar" }) |f|
            if (!@hasField(B.Bank, f)) @compileError(where ++ "Bank has no " ++ f);
        if (@FieldType(B.Bank, "sidecar") != io.UncachedFd) @compileError(where ++ "Bank.sidecar is not an UncachedFd");
        for ([_][]const u8{ "recordOffset", "spans" }) |m|
            if (!@hasDecl(B.Bank, m)) @compileError(where ++ "Bank has no " ++ m);
        if (@hasDecl(B, "scatter") and B.scatter and !@hasDecl(B.Bank, "parts")) @compileError(where ++ "a scatter Bank has no parts");
    }
}

fn expectMethod(comptime S: type, comptime name: []const u8, comptime params: []const type, comptime Payload: type) void {
    const where = @typeName(S) ++ "." ++ name;
    if (!@hasDecl(S, name)) @compileError(@typeName(S) ++ " is not an expert source: no " ++ name);
    const info = @typeInfo(@TypeOf(@field(S, name))).@"fn";
    if (info.param_types.len != params.len) @compileError(where ++ ": the contract takes a different parameter count");
    for (info.param_types, params) |p, t| {
        if (p.? != t) @compileError(where ++ ": parameter " ++ @typeName(p.?) ++ " where the contract has " ++ @typeName(t));
    }
    const R = info.return_type.?;
    const P = switch (@typeInfo(R)) {
        .error_union => |eu| eu.payload,
        else => R,
    };
    if (P != Payload) @compileError(where ++ ": returns " ++ @typeName(P) ++ " where the contract has " ++ @typeName(Payload));
}

const testing = std.testing;

test "sdk expert: Stats.since is each counter's difference, at least zero" {
    const s0: Stats = .{ .route_calls = 3, .expert_bytes_read = 100, .expert_read_seconds = 0.25, .spec_issued = 9 };
    const s1: Stats = .{ .route_calls = 10, .expert_bytes_read = 4196, .expert_read_seconds = 1.0, .spec_issued = 2, .promoted = 5 };
    const d = s1.since(s0);
    try testing.expectEqual(@as(u64, 7), d.route_calls);
    try testing.expectEqual(@as(u64, 4096), d.expert_bytes_read);
    try testing.expectEqual(@as(f64, 0.75), d.expert_read_seconds);
    try testing.expectEqual(@as(u64, 5), d.promoted);
    // A counter below its earlier snapshot (a new stream's) gives zero, not a wrap.
    try testing.expectEqual(@as(u64, 0), d.spec_issued);
    try testing.expectEqual(@as(f64, 0), s0.since(s1).expert_read_seconds);
    try testing.expectEqual(Stats{}, s1.since(s1));
}

test "sdk expert: one reader per process: a second taker is refused by name until the first gives it back" {
    try takeReader();
    try testing.expectError(error.ExpertReaderInUse, takeReader());
    giveReader();
    try takeReader();
    giveReader();
}

test "sdk expert: the reader's topology gate is its per-range limit, for both banks" {
    checkTopology(9, 6); // EXL3
    checkTopology(6, 4); // MXFP4
    checkTopology(12, 6); // within the reader's limits, though no bank uses it
    const exl3: RecordLayout = .{ .components = 9, .gate_up = 6, .segments = undefined, .record_bytes = 13_316_096, .gate_up_bytes = 8_877_056 };
    try testing.expectEqual(@as(usize, 9), exl3.segmentsOf().len);
    var mx: RecordLayout = .{ .components = 6, .gate_up = 4, .segments = undefined, .record_bytes = 0, .gate_up_bytes = 0 };
    // MiMo's MXFP4 record (mxfp4_expert_bank.Geometry at hidden 4096, inter 2048): U32 codes and U8 scales per projection
    const hidden: u64 = 4096;
    const inter: u64 = 2048;
    var off: u64 = 0;
    for (0..3) |p| {
        const in = if (p == 2) inter else hidden;
        const out = if (p == 2) hidden else inter;
        mx.segments[p * 2] = .{ .dtype = .uint32, .shape = .{ out, in / 8, 0 }, .rank = 2, .offset = off, .length = out * in / 2 };
        off += mx.segments[p * 2].length;
        mx.segments[p * 2 + 1] = .{ .dtype = .uint8, .shape = .{ out, in / 32, 0 }, .rank = 2, .offset = off, .length = out * in / 32 };
        off += mx.segments[p * 2 + 1].length;
    }
    mx.record_bytes = off;
    mx.gate_up_bytes = mx.segments[4].offset;
    try testing.expectEqual(@as(usize, 6), mx.segmentsOf().len);
    try testing.expectEqual(mx.gate_up_bytes, mx.segments[0].length + mx.segments[1].length + mx.segments[2].length + mx.segments[3].length);
}
