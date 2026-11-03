//! MTP and DFlash traces and their env switches.
const std = @import("std");
const env = @import("../../../../env.zig");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;


/// Per-phase wall-time accumulator behind diag mtp-trace. Pure
/// bookkeeping; `nextMtp` stamps phases with a Stopwatch and emits one
/// summary line every LOG_EVERY rounds. Zero cost when the env is absent
/// (every stamp is guarded on the cached env check).
pub const MtpTrace = struct {
    pub const LOG_EVERY: u32 = 32;
    pub const Phase = enum(u4) { draft, sync, ext, verify, corr, eval, hist, commit, predraft, gap };
    pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;
    // Sub-laps attribute time inside a phase and are not a partition of it: a group's build lap is
    // stamped into every row while each row's `verify` carries only its own share.
    pub const Sub = enum(u3) { build, ple, ple_sync, dispatch, wait, chain };
    pub const N_SUBS = @typeInfo(Sub).@"enum".field_names.len;

    rounds: u32 = 0,
    ns: [N_PHASES]u64 = @splat(0),
    sub_ns: [N_SUBS]u64 = @splat(0),
    drafted: u64 = 0,
    accepted: u64 = 0,
    extended: u32 = 0,
    /// Per draft index: rounds that drafted index i / accepted it.
    drafted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),
    accepted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),

    pub fn add(self: *MtpTrace, phase: Phase, dur_ns: u64) void {
        self.ns[@backingInt(phase)] += dur_ns;
    }

    pub fn addSub(self: *MtpTrace, sub: Sub, dur_ns: u64) void {
        self.sub_ns[@backingInt(sub)] += dur_ns;
    }

    /// Close one round; true when a summary line is due (caller logs,
    /// then calls reset()).
    pub fn endRound(self: *MtpTrace, drafted_n: u32, accepted_n: u32, was_extended: bool) bool {
        self.rounds += 1;
        self.drafted += drafted_n;
        self.accepted += accepted_n;
        if (was_extended) self.extended += 1;
        var i: usize = 0;
        while (i < drafted_n and i < mtp_mod.MAX_DEPTH) : (i += 1) {
            self.drafted_idx[i] += 1;
            if (i < accepted_n) self.accepted_idx[i] += 1;
        }
        return self.rounds >= LOG_EVERY;
    }

    /// `a0/a1/...` acceptance per drafted index (only indices ever drafted).
    pub fn accIdxStr(self: *const MtpTrace, buf: []u8) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        var i: usize = 0;
        while (i < mtp_mod.MAX_DEPTH and self.drafted_idx[i] > 0) : (i += 1) {
            w.print("{s}{d:.2}", .{ if (i == 0) "" else "/", @as(f64, @floatFromInt(self.accepted_idx[i])) / @as(f64, @floatFromInt(self.drafted_idx[i])) }) catch break;
        }
        return w.buffered();
    }

    pub fn avgMs(self: *const MtpTrace, phase: Phase) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
            (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn subAvgMs(self: *const MtpTrace, sub: Sub) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.sub_ns[@backingInt(sub)])) /
            (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn totalAvgMs(self: *const MtpTrace) f64 {
        if (self.rounds == 0) return 0.0;
        var total: u64 = 0;
        for (self.ns) |v| total += v;
        return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    // Profilers parse the phase summary as one run from `rounds=` to `acc_idx=`; new fields go behind it.
    pub const LINE_FMT = "  [mtp-trace] rounds={d} avg_ms draft={d:.2} sync={d:.2} ext={d:.2} verify={d:.2} corr={d:.2} eval={d:.2} hist={d:.2} commit={d:.2} predraft={d:.2} gap={d:.2} total={d:.2} | m_avg={d:.2} acc_avg={d:.2} ext_rate={d:.2} acc_idx={s} | sub_ms build={d:.2} ple={d:.2} ple_sync={d:.2} dispatch={d:.2} wait={d:.2} chain={d:.2}\n";

    pub fn reset(self: *MtpTrace) void {
        self.* = .{};
    }
};

pub const SubLap = struct {
    watch: ?io_util.Stopwatch,

    pub fn start(on: bool, io: std.Io) SubLap {
        return .{ .watch = if (on) io_util.Stopwatch.init(io) else null };
    }

    pub fn read(self: SubLap) ?u64 {
        return if (self.watch) |w| w.read() else null;
    }

    pub fn stop(self: SubLap, trace: *MtpTrace, sub: MtpTrace.Sub) void {
        if (self.read()) |ns| trace.addSub(sub, ns);
    }
};

/// Per-phase wall-time accumulator behind diag dflash-trace.
/// `nextDflash` stamps phases with a Stopwatch and emits one summary line
/// every LOG_EVERY rounds. Zero cost when the env is absent.
pub const DflashTrace = struct {
    pub const LOG_EVERY: u32 = 16;
    pub const Phase = enum(u4) { assist, head, verify, accept, append, gap };
    pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

    rounds: u32 = 0,
    ns: [N_PHASES]u64 = @splat(0),
    accepted: u64 = 0,

    pub fn add(self: *DflashTrace, phase: Phase, dur_ns: u64) void {
        self.ns[@backingInt(phase)] += dur_ns;
    }

    /// Close one round; true when a summary line is due (caller logs,
    /// then calls reset()).
    pub fn endRound(self: *DflashTrace, accepted_n: u32) bool {
        self.rounds += 1;
        self.accepted += accepted_n;
        return self.rounds >= LOG_EVERY;
    }

    pub fn avgMs(self: *const DflashTrace, phase: Phase) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
            (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn totalAvgMs(self: *const DflashTrace) f64 {
        if (self.rounds == 0) return 0.0;
        var total: u64 = 0;
        for (self.ns) |v| total += v;
        return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn reset(self: *DflashTrace) void {
        self.* = .{};
    }
};

var dflash_trace_cache: ?bool = null;
pub fn dflashTraceEnabled() bool {
    if (dflash_trace_cache) |v| return v;
    const on = env.one(.DFLASH_TRACE);
    dflash_trace_cache = on;
    return on;
}

/// Close one traced dflash round; emits + resets at the cadence.
pub fn dflashTraceRoundEnd(self: *Generator, accepted: u32) void {
    if (!self.dflash_trace.endRound(accepted)) return;
    const t = &self.dflash_trace;
    log.info(
        "  [dflash-trace] rounds={d} avg_ms assist={d:.2} head={d:.2} verify={d:.2} accept={d:.2} append={d:.2} gap={d:.2} total={d:.2} | acc_avg={d:.2}\n",
        .{
            t.rounds,
            t.avgMs(.assist),
            t.avgMs(.head),
            t.avgMs(.verify),
            t.avgMs(.accept),
            t.avgMs(.append),
            t.avgMs(.gap),
            t.totalAvgMs(),
            @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
        },
    );
    t.reset();
}

/// Cross-round pre-draft (round pipelining): the next round's chunk-A draft chain is built and dispatched at the
/// current round's tail (see mtpMaybePreDraft) so the GPU drafts while the CPU does emit/SSE bookkeeping. Test seam:
/// false observes a completed round before any successor consumes it.
pub var mtp_predraft_override: ?bool = null;
pub fn mtpPredraftEnabled() bool {
    return mtp_predraft_override orelse true;
}

pub fn mtpTraceOn() bool {
    return mtpTraceEnabled();
}

pub fn mtpTraceSub(self: *Generator, sub: MtpTrace.Sub, dur_ns: u64) void {
    if (!mtpTraceEnabled()) return;
    self.mtp_trace.addSub(sub, dur_ns);
}

var mtp_trace_cache: ?bool = null;
pub fn mtpTraceEnabled() bool {
    if (mtp_trace_cache) |v| return v;
    const on = env.one(.MTP_TRACE);
    mtp_trace_cache = on;
    return on;
}

// ── tests ──


test "MtpTrace: per-phase accumulation, round averaging, log cadence, reset" {
    var t = Generator.MtpTrace{};
    // 2 rounds: draft 2ms+4ms, eval 10ms+30ms.
    t.add(.draft, 2_000_000);
    t.add(.eval, 10_000_000);
    try testing.expect(!t.endRound(3, 2, false));
    t.add(.draft, 4_000_000);
    t.add(.eval, 30_000_000);
    try testing.expect(!t.endRound(7, 6, true));
    try testing.expectApproxEqAbs(@as(f64, 3.0), t.avgMs(.draft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.eval), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 23.0), t.totalAvgMs(), 1e-9);
    try testing.expectEqual(@as(u64, 10), t.drafted);
    try testing.expectEqual(@as(u64, 8), t.accepted);
    try testing.expectEqual(@as(u32, 1), t.extended);
    // Log line falls due exactly at LOG_EVERY rounds.
    var i: u32 = 2;
    while (i < Generator.MtpTrace.LOG_EVERY - 1) : (i += 1) {
        try testing.expect(!t.endRound(1, 1, false));
    }
    try testing.expect(t.endRound(1, 1, false));
    t.reset();
    try testing.expectEqual(@as(u32, 0), t.rounds);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.draft), 1e-9);
}

test "the [mtp-trace] phase summary stays contiguous for its log consumers" {
    const line = try std.fmt.allocPrint(testing.allocator, Generator.MtpTrace.LINE_FMT, .{
        @as(u32, 32), 0.45, 0.0, 0.0,         4.09, 56.86, 0.0,  0.0,   0.03, 0.0,  0.07, 61.57,
        2.0,          1.4,  0.0, "0.91/0.80", 4.09, 7.13,  6.40, 52.28, 0.94, 0.44,
    });
    defer testing.allocator.free(line);
    try testing.expect(std.mem.indexOf(u8, line, "total=61.57 | m_avg=") != null);
    const acc_at = std.mem.indexOf(u8, line, "acc_idx=").?;
    const sub_at = std.mem.indexOf(u8, line, "| sub_ms ").?;
    try testing.expect(acc_at < sub_at);
}

test "MtpTrace sub-laps attribute inside a phase without changing the round total" {
    var t = Generator.MtpTrace{};
    t.add(.verify, 20_000_000);
    t.addSub(.build, 15_000_000);
    t.addSub(.ple, 3_000_000);
    t.addSub(.ple_sync, 2_000_000);
    t.add(.corr, 6_000_000);
    t.addSub(.dispatch, 2_000_000);
    t.add(.eval, 60_000_000);
    t.addSub(.wait, 58_000_000);
    t.add(.gap, 12_000_000);
    t.addSub(.chain, 2_000_000);
    try testing.expect(!t.endRound(2, 2, false));
    try testing.expectApproxEqAbs(@as(f64, 98.0), t.totalAvgMs(), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.verify), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.subAvgMs(.build) + t.subAvgMs(.ple) + t.subAvgMs(.ple_sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 58.0), t.subAvgMs(.wait), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), t.subAvgMs(.chain), 1e-9);
    t.reset();
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.subAvgMs(.build), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.totalAvgMs(), 1e-9);
}

test "MtpTrace per-index acceptance: index i counts only rounds that drafted it" {
    var t: Generator.MtpTrace = .{};
    _ = t.endRound(3, 3, false); // all three accepted
    _ = t.endRound(3, 1, false); // a0 accepted, a1 rejected, a2 never judged
    _ = t.endRound(1, 0, false); // depth-1 round: only index 0 drafted
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("0.67/0.50/0.50", t.accIdxStr(&buf));
    t.reset();
    try std.testing.expectEqualStrings("", t.accIdxStr(&buf));
}
