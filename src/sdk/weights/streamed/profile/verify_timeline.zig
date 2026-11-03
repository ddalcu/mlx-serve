//! The verify's GPU timeline, for PROFILE builds only (`dsv41_decode_timers.enabled`; in every other build each entry
//! below compiles to nothing and the shim, lib/expert_io/dsv41_cb_timeline.mm, is not linked). Active in a
//! decode-profile run (`install`): every command buffer MLX's GPU queue commits during the decode is recorded with
//! its host commit time, Metal's GPUStartTime / GPUEndTime and its completion handler's host time; each verify's
//! window and, per routed layer, the host stamps of the layer's submits (`Point`). All storage is static; the
//! summary (`VERIFY_GPU_TIMELINE`) and the raw rows (`writeJson`) are made after the decode.
const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const dt = @import("decode_timers.zig");

pub const enabled = dt.enabled;

/// Set by `install`, cleared by `uninstall`.
pub var active: bool = false;

/// Cycles <= max_tokens - 1 (the cell's default cap: 1,024 ids).
pub const max_cycles = 1024;
pub const max_layers = @import("decode_timers.zig").max_layers;
/// Buffers past this are counted (`dropped`), not recorded.
pub const max_buffers = 1 << 18;

/// One committed command buffer (the shim's `Dsv41tlRow`). Times: mach_absolute_time ns.
/// `queue`: 1 for the queue of the buffer the install saw (MLX's stream), 0 for another queue.
pub const Row = extern struct { host_commit: u64 = 0, gpu_start: u64 = 0, gpu_end: u64 = 0, host_done: u64 = 0, tag: u64 = 0, status: u64 = 0, queue: u64 = 0 };

pub const Phase = enum(u8) { draft = 1, verify = 2, rest = 3 };

/// A routed layer's host stamps in a verify (`Experts.run`): the barrier's eval submitted (`call`), the ids read
/// (`barrier`), the route planned (`route`), the hit wave and hoist submitted (`hit`), the call's last submit (`end`).
pub const Point = enum(u8) { call, barrier, route, hit, end };
/// `gate_gu` / `gate_down`: the stream event's values the layer's gated miss parts wait for (its gate/up value and its
/// last down part's), 0 without a gated part.
pub const Layer = struct { t: [5]u64 = @splat(0), read_call: u64 = 0, read_end: u64 = 0, missed: bool = false, gate_gu: u64 = 0, gate_down: u64 = 0 };
pub const Verify = struct { begin: u64 = 0, end: u64 = 0 };

/// The read pool's event-signal log (`q3ld_test_ev_log`, compiled in the profile and test builds): per signal the
/// value handed to the event, the host time of the signal call, the time the satisfied prefix advanced (the bytes
/// landed) and the call's ns. Past the cap the pool stops logging (`signals_full`).
pub const max_signals = 1 << 17;
pub const Signal = [4]i64;

pub const storage_bytes = max_buffers * @sizeOf(Row) + max_cycles * (max_layers * @sizeOf(Layer) + @sizeOf(Verify)) + max_signals * @sizeOf(Signal);

var rows: [max_buffers]Row = undefined;
var layers: [max_cycles][max_layers]Layer = undefined;
var verifies: [max_cycles]Verify = undefined;
var signals: [max_signals]Signal = undefined;
var n_signals: usize = 0;
/// Cycles begun (`cycleBegin`), and the one in progress.
pub var cycles: u32 = 0;
var cur: u32 = 0;
var in_verify = false;
pub var n_layers: u32 = 0;
var t0: u64 = 0;
/// The stream whose current buffer every cycle's begin re-checks (a buffer of an unhooked class is hooked).
var stream: ?mlx.mlx_stream = null;

const c = if (enabled) struct {
    pub extern fn dsv41tl_mlx_buffer(s: mlx.mlx_stream) ?*anyopaque;
    pub extern fn dsv41tl_install(buf: ?*anyopaque, rows: [*]Row, cap: u32) c_int;
    pub extern fn dsv41tl_tag(tag: u64) void;
    pub extern fn dsv41tl_counts(out: *[3]u32) void;
    pub extern fn dsv41tl_rehook(buf: ?*anyopaque) c_int;
    pub extern fn dsv41tl_hook_stats(out: *[4]u64) void;
    pub extern fn dsv41tl_hooked_names(out: [*]u8, cap: usize) usize;
    pub extern fn dsv41tl_selftest(rows: [*]Row, cap: u32) c_int;
    pub extern fn q3ld_test_ev_log(buf: ?[*]i64, cap: i64) i64;
} else struct {};
// The guards below (`@hasDecl(c, ...)`) see only pub declarations: a profile build must reach every shim call.
comptime {
    if (enabled) for (.{ "dsv41tl_tag", "dsv41tl_rehook", "dsv41tl_counts", "dsv41tl_hook_stats", "dsv41tl_hooked_names", "q3ld_test_ev_log" }) |n| {
        if (!@hasDecl(c, n)) @compileError("dsv41 verify timeline: the shim's " ++ n ++ " is not reachable");
    };
}

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Hooks MLX's command buffers on `s` and arms the storage; refuses what it cannot hold or hook (before the decode).
pub fn install(s: mlx.mlx_stream, max_tokens: u32, layer_count: u32) !void {
    if (comptime !enabled) return;
    try bound(max_tokens, layer_count);
    const rc = c.dsv41tl_install(c.dsv41tl_mlx_buffer(s), &rows, max_buffers);
    if (rc != 0) return error.TimelineNoCommandBuffer;
    stream = s;
    _ = c.q3ld_test_ev_log(@ptrCast(&signals), max_signals);
    arm(layer_count);
}

pub fn bound(max_tokens: u32, layer_count: u32) !void {
    if (max_tokens == 0 or max_tokens - 1 > max_cycles) return error.TimelineCycles;
    if (layer_count == 0 or layer_count > max_layers) return error.TimelineLayers;
}

/// The host side without the shim (no command buffer is recorded): the loop's stamps only (tests).
pub fn armHost(layer_count: u32) void {
    if (comptime !enabled) return;
    arm(layer_count);
}

pub fn verifyOf(ci: usize) Verify {
    return verifies[ci];
}

pub fn layersOf(ci: usize) []const Layer {
    return layers[ci][0..n_layers];
}

fn arm(layer_count: u32) void {
    n_layers = layer_count;
    cycles = 0;
    cur = 0;
    in_verify = false;
    n_signals = 0;
    t0 = now();
    active = true;
}

fn tag(p: Phase) void {
    if (comptime @hasDecl(c, "dsv41tl_tag")) c.dsv41tl_tag((@as(u64, cur) + 1) << 8 | @intFromEnum(p));
}

/// A DSpark cycle begins (its draft block).
pub fn cycleBegin() void {
    if (comptime !enabled) return;
    if (!active) return;
    std.debug.assert(cycles < max_cycles);
    cur = cycles;
    cycles += 1;
    verifies[cur] = .{};
    layers[cur] = @splat(.{});
    if (comptime @hasDecl(c, "dsv41tl_rehook")) if (stream) |s| {
        _ = c.dsv41tl_rehook(c.dsv41tl_mlx_buffer(s));
    };
    tag(.draft);
}

pub fn verifyBegin() void {
    if (comptime !enabled) return;
    if (!active or cycles == 0) return;
    verifies[cur].begin = now();
    in_verify = true;
    tag(.verify);
}

pub fn verifyEnd() void {
    if (comptime !enabled) return;
    if (!active or !in_verify) return;
    verifies[cur].end = now();
    in_verify = false;
    tag(.rest);
}

/// A routed layer's stamp in the verify; `read_gauge` is the stream's read-busy gauge (ns) at `call` and `end`.
pub fn point(layer: u32, p: Point, read_gauge: u64, missed: bool) void {
    if (comptime !enabled) return;
    if (!active or !in_verify or layer >= max_layers) return;
    const l = &layers[cur][layer];
    l.t[@intFromEnum(p)] = now();
    switch (p) {
        .call => l.read_call = read_gauge,
        .end => {
            l.read_end = read_gauge;
            l.missed = missed;
        },
        else => {},
    }
}

/// A routed layer's gated miss parts in the verify: the event values its gate/up waves and its last down wave wait for.
pub fn gate(layer: u32, gu: u64, down_first: u64, n_parts: u32) void {
    if (comptime !enabled) return;
    if (!active or !in_verify or layer >= max_layers or n_parts == 0) return;
    layers[cur][layer].gate_gu = gu;
    layers[cur][layer].gate_down = down_first + n_parts - 1;
}

/// Stops recording (the decode has ended); every buffer committed after is not a row.
pub fn disarm() void {
    if (comptime !enabled) return;
    if (comptime @hasDecl(c, "dsv41tl_tag")) c.dsv41tl_tag(0);
}

/// The hook's own counters: every override entry, the tagged ones, the tagged ones of another queue, the classes hooked.
pub fn hookStats() [4]u64 {
    var out: [4]u64 = @splat(0);
    if (comptime @hasDecl(c, "dsv41tl_hook_stats")) c.dsv41tl_hook_stats(&out);
    return out;
}

/// Before the first cycle (the phase change committed buffers after the install, at the prompt's end): why the GPU
/// timeline cannot run (the hook saw no commit), else null. The cell then does not install it for the decode.
pub fn unavailableReason() ?[]const u8 {
    if (comptime !enabled) return null;
    if (!active) return null;
    if (hookStats()[0] == 0) return "the hook saw no commit of the phase change";
    return null;
}

/// `VERIFY_GPU_TIMELINE_UNAVAILABLE {...}`: the reason and what the hook saw (its counters and classes).
pub fn unavailableLine(buf: []u8, reason: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    const hs = hookStats();
    var names: [512]u8 = undefined;
    const nlen: usize = if (comptime @hasDecl(c, "dsv41tl_hooked_names")) c.dsv41tl_hooked_names(&names, names.len) else 0;
    w.print("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"{s}\", \"calls\": {d}, \"tagged\": {d}, \"other_queue\": {d}, \"classes\": {d}, \"names\": \"{s}\"}}", .{ reason, hs[0], hs[1], hs[2], hs[3], names[0..nlen] }) catch return buf[0..0];
    return w.buffered();
}

pub fn uninstall() void {
    if (comptime !enabled) return;
    disarm();
    if (comptime @hasDecl(c, "q3ld_test_ev_log")) _ = c.q3ld_test_ev_log(null, -1);
    active = false;
}

/// `Row.status` bit set on a row `repairRows` gave a start (Metal statuses are <= 5).
pub const repaired_status: u64 = 1 << 8;
pub const Repair = struct { unstarted: u32 = 0, repaired: u32 = 0, zero_length: u32 = 0 };

/// Rows Metal completed with GPUStartTime 0 and a GPUEndTime: a buffer that ran no kernel and only waited (an event
/// wait in its own buffer, as MiMo's gated miss parts commit them). Read raw, such a row spans [0, end): the busy union
/// takes the verify from its begin and the gated wait from time 0 (the `bytes_wait_ms` ~1e10 of the first MiMo MTP
/// cells). Its start is the queue's: MLX's stream queue runs its buffers in commit order (the row order), so the GPU
/// reached it at the later of its commit and the previous row's end. Set there and marked `repaired_status`; a row
/// whose queue start is not before its end keeps start 0 (dropped by every span, counted `unstarted` only). Pure over
/// `rs` (the MiMo profile calls it on the stored rows after `settle`; DSV's cell does not call it).
pub fn repairRows(rs: []Row) Repair {
    var out: Repair = .{};
    var prev_end: u64 = 0;
    for (rs) |*r| {
        if (r.queue != 1) continue;
        if (r.gpu_start == 0 and r.gpu_end != 0) {
            out.unstarted += 1;
            const start = @max(r.host_commit, prev_end);
            if (start < r.gpu_end) {
                r.gpu_start = start;
                r.status |= repaired_status;
                out.repaired += 1;
            }
        } else if (r.gpu_end != 0 and r.gpu_end == r.gpu_start) out.zero_length += 1;
        prev_end = @max(prev_end, r.gpu_end);
    }
    return out;
}

/// `repairRows` over the stored rows (after `settle`); nothing outside profile builds.
pub fn repairUnstarted() Repair {
    if (comptime !enabled) return .{};
    return repairRows(rows[0..@min(counts()[0], max_buffers)]);
}

/// The stored rows, verifies and layer stamps (read after `settle`), for a plugin's own summary over them; empty
/// outside profile builds. DSV's cell does not call it (the MiMo MTP profile's `MIMO_BARRIER_RESIDUAL` does).
pub const Stored = struct { rows: []const Row, verifies: []const Verify, layers: []const [max_layers]Layer, n_layers: u32 };

pub fn stored() Stored {
    if (comptime !enabled) return .{ .rows = &.{}, .verifies = &.{}, .layers = &.{}, .n_layers = 0 };
    return .{ .rows = rows[0..@min(counts()[0], max_buffers)], .verifies = verifies[0..cycles], .layers = layers[0..cycles], .n_layers = n_layers };
}

/// A row's cycle (null: untagged) and phase (`Phase`'s value), as the summary reads its tag.
pub fn rowCycle(r: Row) ?u32 {
    return tagCycle(r.tag);
}

pub fn rowPhase(r: Row) u8 {
    return tagPhase(r.tag);
}

/// The tag `cycleBegin` / `verifyBegin` give the buffers of cycle `cyc`'s phase `p` (hand-made rows in tests).
pub fn tagOf(cyc: u32, p: Phase) u64 {
    return (@as(u64, cyc) + 1) << 8 | @intFromEnum(p);
}

/// (committed, completed, dropped) from the shim.
pub fn counts() [3]u32 {
    var out: [3]u32 = .{ 0, 0, 0 };
    if (comptime @hasDecl(c, "dsv41tl_counts")) c.dsv41tl_counts(&out);
    return out;
}

/// After the decode: syncs `s` and waits (at most `timeout_ms`) for every recorded buffer's handler. Returns the
/// rows still pending.
pub fn settle(s: mlx.mlx_stream, timeout_ms: u64) u32 {
    if (comptime !enabled) return 0;
    disarm();
    _ = mlx.mlx_synchronize(s);
    n_signals = @intCast(@min(c.q3ld_test_ev_log(null, 0), max_signals));
    const start = now();
    while (true) {
        const k = counts();
        const want = @min(k[0], max_buffers);
        if (k[1] >= want) return 0;
        if (now() - start > timeout_ms * std.time.ns_per_ms) return want - k[1];
        std.Thread.yield() catch {};
    }
}

// ── the summary (pure over the stored rows; the tests feed it by hand) ──

fn tagCycle(t: u64) ?u32 {
    if (t == 0) return null;
    return @intCast((t >> 8) - 1);
}

fn tagPhase(t: u64) u8 {
    return @truncate(t);
}

/// A host- or GPU-clock interval [start, end) (the prompt timeline's too, `mimo_v2/prefill_timeline.zig`).
pub const Span = [2]u64;

/// The busy union of the cycle's buffers inside [lo, hi), merged and sorted (into `out`).
fn busyUnion(rs: []const Row, cyc: u32, lo: u64, hi: u64, out: []Span) []Span {
    var n: usize = 0;
    for (rs) |r| {
        if (tagCycle(r.tag) != cyc or r.gpu_end <= r.gpu_start) continue;
        const a = @max(r.gpu_start, lo);
        const b = @min(r.gpu_end, hi);
        if (b <= a or n == out.len) continue;
        out[n] = .{ a, b };
        n += 1;
    }
    std.mem.sort(Span, out[0..n], {}, struct {
        fn lt(_: void, x: Span, y: Span) bool {
            return x[0] < y[0];
        }
    }.lt);
    var m: usize = 0;
    for (out[0..n]) |sp| {
        if (m > 0 and sp[0] <= out[m - 1][1]) {
            out[m - 1][1] = @max(out[m - 1][1], sp[1]);
        } else {
            out[m] = sp;
            m += 1;
        }
    }
    return out[0..m];
}

/// The busy time of `u` inside [a, b).
pub fn busyIn(u: []const Span, a: u64, b: u64) u64 {
    var s: u64 = 0;
    for (u) |sp| {
        const x = @max(sp[0], a);
        const y = @min(sp[1], b);
        if (y > x) s += y - x;
    }
    return s;
}

fn idleIn(u: []const Span, a: u64, b: u64) u64 {
    if (b <= a) return 0;
    return (b - a) - busyIn(u, a, b);
}

/// The first logged signal at or past `value` (the log's values never decrease): its signal-call time.
pub fn signalOf(sig: []const Signal, value: u64) ?u64 {
    var lo: usize = 0;
    var hi: usize = sig.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (@as(u64, @intCast(sig[mid][0])) < value) lo = mid + 1 else hi = mid;
    }
    return if (lo < sig.len) @intCast(sig[lo][1]) else null;
}

/// The GPU span (first start, last end) of the cycle's verify buffers committed in [a, b] (host time).
fn spanOf(rs: []const Row, cyc: u32, a: u64, b: u64) ?Span {
    var sp: ?Span = null;
    for (rs) |r| {
        if (tagCycle(r.tag) != cyc or tagPhase(r.tag) != @intFromEnum(Phase.verify) or r.host_commit < a or r.host_commit > b or r.gpu_end <= r.gpu_start) continue;
        sp = if (sp) |x| .{ @min(x[0], r.gpu_start), @max(x[1], r.gpu_end) } else .{ r.gpu_start, r.gpu_end };
    }
    return sp;
}

/// The GPU end of the last verify buffer committed in [a, b] (host time): the routing barrier's buffer.
fn lastEnd(rs: []const Row, cyc: u32, a: u64, b: u64) ?u64 {
    var e: ?u64 = null;
    var t: u64 = 0;
    for (rs) |r| {
        if (tagCycle(r.tag) != cyc or tagPhase(r.tag) != @intFromEnum(Phase.verify) or r.host_commit < a or r.host_commit > b or r.gpu_end <= r.gpu_start) continue;
        if (e == null or r.host_commit >= t) {
            e = r.gpu_end;
            t = r.host_commit;
        }
    }
    return e;
}

/// Per routed layer (ns): `idle_barrier` (GPU idle from the barrier's submit to the hit wave's), `hit_span` (the hit
/// wave's buffers, first GPU start to last end), `gap` (the routing barrier's buffer end to the hit wave's first start),
/// `wait` / `wait_gu` (the layer's gated miss parts: from the GPU start of the first buffer after its hit wave to the
/// signal of its last down value / of its gate-up value, capped at those buffers' end).
pub const PerLayer = struct { idle_barrier: u64 = 0, hit_span: ?u64 = null, gap: ?u64 = null, wait: ?u64 = null, wait_gu: ?u64 = null };

/// One verify (ns): its wall, the GPU's busy union inside it and the idle rest, split by the host's stamps into
/// `idle_barrier`, `idle_encode` (before the first layer, and between a layer's hit submit and the next layer's barrier
/// submit) and `idle_tail` (after the last layer's hit submit); `read` the stream's read gauge over the routed calls;
/// `miss_busy` the GPU busy time after the hit submit of the layers that missed; `wait` the summed gated waits on miss
/// bytes (`PerLayer.wait`) and `work` = busy - wait; `hit_span` / `gap` summed over the layers that have them.
pub const CycleSum = struct { wall: u64 = 0, busy: u64 = 0, idle: u64 = 0, idle_barrier: u64 = 0, idle_encode: u64 = 0, idle_tail: u64 = 0, read: u64 = 0, miss_busy: u64 = 0, buffers: u32 = 0, layers: u32 = 0, truncated: bool = false, work: u64 = 0, wait: u64 = 0, wait_gu: u64 = 0, hit_span: u64 = 0, hit_layers: u32 = 0, gap: u64 = 0, gap_layers: u32 = 0, gated: u32 = 0, unsignalled: u32 = 0 };

pub fn cycleSum(rs: []const Row, v: Verify, ls: []const Layer, cyc: u32, sig: []const Signal, scratch: []Span, per: []PerLayer) CycleSum {
    var s: CycleSum = .{};
    for (per) |*p| p.* = .{};
    if (v.end <= v.begin) return s;
    s.wall = v.end - v.begin;
    for (rs) |r| s.buffers += @intFromBool(tagCycle(r.tag) == cyc and tagPhase(r.tag) == @intFromEnum(Phase.verify));
    const u = busyUnion(rs, cyc, v.begin, v.end, scratch);
    s.truncated = u.len == scratch.len;
    s.busy = busyIn(u, v.begin, v.end);
    s.idle = s.wall - s.busy;
    var prev_hit: u64 = v.begin;
    var first: ?usize = null;
    var last: ?usize = null;
    const P = struct {
        fn t(l: Layer, p: Point) u64 {
            return l.t[@intFromEnum(p)];
        }
    };
    for (ls, 0..) |l, i| {
        if (P.t(l, .call) == 0) continue;
        const call = P.t(l, .call);
        const hit = P.t(l, .hit);
        s.idle_encode += idleIn(u, prev_hit, call);
        per[i].idle_barrier = idleIn(u, call, hit);
        s.idle_barrier += per[i].idle_barrier;
        const hs = spanOf(rs, cyc, P.t(l, .route), hit);
        if (hs) |h| {
            per[i].hit_span = h[1] - h[0];
            s.hit_span += h[1] - h[0];
            s.hit_layers += 1;
            if (lastEnd(rs, cyc, call, P.t(l, .barrier))) |re| {
                per[i].gap = h[0] -| re;
                s.gap += h[0] -| re;
                s.gap_layers += 1;
            }
        }
        prev_hit = hit;
        s.layers += 1;
        if (first == null) first = i;
        last = i;
    }
    s.idle_tail = idleIn(u, prev_hit, v.end);
    if (first) |f| s.read = ls[last.?].read_end -| ls[f].read_call;
    for (ls, 0..) |l, i| {
        if (P.t(l, .call) == 0) continue;
        // the next stamped layer's barrier (its eval commits the miss waves), or the verify's end
        var next_call: u64 = v.end;
        var next_barrier: u64 = v.end;
        for (ls[i + 1 ..]) |n| if (P.t(n, .call) != 0) {
            next_call = P.t(n, .call);
            next_barrier = P.t(n, .barrier);
            break;
        };
        if (l.missed) s.miss_busy += busyIn(u, P.t(l, .hit), next_call);
        if (l.gate_down == 0) continue;
        s.gated += 1;
        const w = spanOf(rs, cyc, P.t(l, .hit) + 1, next_barrier) orelse continue;
        const b = if (per[i].hit_span != null) @max(w[0], (spanOf(rs, cyc, P.t(l, .route), P.t(l, .hit)).?)[1]) else w[0];
        const sd = signalOf(sig, l.gate_down) orelse {
            s.unsignalled += 1;
            continue;
        };
        const wait = @min(sd, w[1]) -| b;
        per[i].wait = wait;
        s.wait += wait;
        if (signalOf(sig, l.gate_gu)) |sg| {
            per[i].wait_gu = @min(sg, w[1]) -| b;
            s.wait_gu += per[i].wait_gu.?;
        }
    }
    s.work = s.busy -| s.wait;
    return s;
}

const fields = [_][]const u8{ "wall_ms", "busy_ms", "gpu_work_ms", "bytes_wait_ms", "bytes_wait_gu_ms", "idle_ms", "idle_barrier_ms", "idle_barrier_per_layer_ms", "idle_encode_ms", "idle_tail_ms", "hit_span_ms", "hit_span_per_layer_ms", "router_hit_gap_ms", "router_hit_gap_per_layer_ms", "read_busy_ms", "miss_layer_busy_ms", "buffers", "gated_layers", "unsignalled_gates" };

fn fieldOf(s: CycleSum, i: usize) f64 {
    const ms = struct {
        fn f(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6;
        }
    }.f;
    const per = struct {
        fn f(x: u64, n: u32) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6 / @as(f64, @floatFromInt(@max(n, 1)));
        }
    }.f;
    return switch (i) {
        0 => ms(s.wall),
        1 => ms(s.busy),
        2 => ms(s.work),
        3 => ms(s.wait),
        4 => ms(s.wait_gu),
        5 => ms(s.idle),
        6 => ms(s.idle_barrier),
        7 => per(s.idle_barrier, s.layers),
        8 => ms(s.idle_encode),
        9 => ms(s.idle_tail),
        10 => ms(s.hit_span),
        11 => per(s.hit_span, s.hit_layers),
        12 => ms(s.gap),
        13 => per(s.gap, s.gap_layers),
        14 => ms(s.read),
        15 => ms(s.miss_busy),
        16 => @floatFromInt(s.buffers),
        17 => @floatFromInt(s.gated),
        else => @floatFromInt(s.unsignalled),
    };
}

fn median(xs: []f64) f64 {
    if (xs.len == 0) return 0;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

const by_layer_fields = [_][]const u8{ "idle_barrier", "hit_span", "router_hit_gap", "bytes_wait" };

var per_cycle: [fields.len][max_cycles]f64 = undefined;
var per_layer: [by_layer_fields.len][max_layers][max_cycles]f64 = undefined;
var per_layer_n: [by_layer_fields.len][max_layers]usize = undefined;
var span_scratch: [8192]Span = undefined;

/// `VERIFY_GPU_TIMELINE {...}` over the stored verifies: per-cycle medians and means, per-layer medians (over the
/// cycles where the layer has the quantity), the buffer and signal counts (`pending`: rows whose handler had not run
/// at `settle`).
pub fn line(buf: []u8, pending: u32) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    writeLine(&w, rows[0..@min(counts()[0], max_buffers)], verifies[0..cycles], layers[0..cycles], signals[0..n_signals], pending) catch return buf[0..0];
    return w.buffered();
}

fn writeLine(w: *std.Io.Writer, rs: []const Row, vs: []const Verify, ls: []const [max_layers]Layer, sig: []const Signal, pending: u32) !void {
    var n: usize = 0;
    var truncated: u32 = 0;
    var verify_buffers: u64 = 0;
    var per: [max_layers]PerLayer = undefined;
    per_layer_n = @splat(@splat(0));
    for (vs, 0..) |v, ci| {
        if (v.end <= v.begin) continue;
        const s = cycleSum(rs, v, ls[ci][0..n_layers], @intCast(ci), sig, &span_scratch, per[0..n_layers]);
        for (0..fields.len) |f| per_cycle[f][n] = fieldOf(s, f);
        for (per[0..n_layers], 0..) |p, l| {
            const vals = [_]?u64{ p.idle_barrier, p.hit_span, p.gap, p.wait };
            for (vals, 0..) |x, k| if (x) |y| {
                per_layer[k][l][per_layer_n[k][l]] = @as(f64, @floatFromInt(y)) / 1e6;
                per_layer_n[k][l] += 1;
            };
        }
        truncated += @intFromBool(s.truncated);
        verify_buffers += s.buffers;
        n += 1;
    }
    const k = counts();
    const hs = hookStats();
    var other_rows: u64 = 0;
    for (rs) |r| other_rows += @intFromBool(r.queue == 0);
    var names: [512]u8 = undefined;
    const nlen: usize = if (comptime @hasDecl(c, "dsv41tl_hooked_names")) c.dsv41tl_hooked_names(&names, names.len) else 0;
    try w.print("VERIFY_GPU_TIMELINE {{\"cycles\": {d}, \"verifies\": {d}, \"layers\": {d}, \"buffers\": {d}, \"verify_buffers\": {d}, \"dropped\": {d}, \"pending\": {d}, \"truncated_cycles\": {d}, \"signals\": {d}, \"signals_full\": {}, \"hook\": {{\"calls\": {d}, \"tagged\": {d}, \"other_queue\": {d}, \"other_queue_rows\": {d}, \"classes\": {d}, \"names\": \"{s}\"}}, \"clock\": \"mach_absolute ns; GPUStartTime / GPUEndTime per committed MTLCommandBuffer; bytes_wait to the read pool's host call of setSignaledValue\"", .{ cycles, n, n_layers, rs.len, verify_buffers, k[2], pending, truncated, sig.len, sig.len >= max_signals, hs[0], hs[1], hs[2], other_rows, hs[3], names[0..nlen] });
    inline for (.{ "median", "mean" }) |kind| {
        try w.print(", \"{s}\": {{", .{kind});
        for (fields, 0..) |name, f| {
            const xs = per_cycle[f][0..n];
            var v: f64 = 0;
            if (comptime std.mem.eql(u8, kind, "mean")) {
                for (xs) |x| v += x;
                v /= @floatFromInt(@max(n, 1));
            } else v = median(xs);
            try w.print("{s}\"{s}\": {d:.3}", .{ if (f == 0) "" else ", ", name, v });
        }
        try w.writeAll("}");
    }
    for (by_layer_fields, 0..) |name, f| {
        try w.print(", \"{s}_ms_by_layer_median\": [", .{name});
        for (0..n_layers) |l| try w.print("{s}{d:.3}", .{ if (l == 0) "" else ", ", median(per_layer[f][l][0..per_layer_n[f][l]]) });
        try w.writeAll("]");
    }
    try w.writeAll("}");
}

/// The receipt's `verify_gpu_timeline`: every recorded buffer, every verify's stamps and the pool's signals, in us from
/// the install.
pub fn writeJson(w: *std.Io.Writer) !void {
    const rs = rows[0..@min(counts()[0], max_buffers)];
    const us = struct {
        fn f(x: u64) f64 {
            return if (x == 0) -1 else (@as(f64, @floatFromInt(x)) - @as(f64, @floatFromInt(t0))) / 1e3;
        }
    }.f;
    try w.writeAll("{\"time\": \"us from the install (mach_absolute); -1 = not stamped\", \"buffer_fields\": [\"cycle\", \"phase(1 draft, 2 verify, 3 rest)\", \"host_commit\", \"gpu_start\", \"gpu_end\", \"host_done\", \"status\"], \"buffers\": [");
    for (rs, 0..) |r, i| {
        const cyc: i64 = if (tagCycle(r.tag)) |x| x else -1;
        try w.print("{s}[{d}, {d}, {d:.1}, {d:.1}, {d:.1}, {d:.1}, {d}]", .{ if (i == 0) "" else ", ", cyc, tagPhase(r.tag), us(r.host_commit), us(r.gpu_start), us(r.gpu_end), us(r.host_done), r.status });
    }
    try w.writeAll("], \"layer_fields\": [\"layer\", \"call\", \"barrier\", \"route\", \"hit\", \"end\", \"read_gauge_call_us\", \"read_gauge_end_us\", \"missed\", \"gate_gu\", \"gate_down\"], \"verifies\": [");
    for (verifies[0..cycles], 0..) |v, ci| {
        try w.print("{s}{{\"cycle\": {d}, \"begin\": {d:.1}, \"end\": {d:.1}, \"layers\": [", .{ if (ci == 0) "" else ", ", ci, us(v.begin), us(v.end) });
        var first = true;
        for (layers[ci][0..n_layers], 0..) |l, li| {
            if (l.t[0] == 0) continue;
            try w.print("{s}[{d}", .{ if (first) "" else ", ", li });
            for (l.t) |t| try w.print(", {d:.1}", .{us(t)});
            try w.print(", {d:.1}, {d:.1}, {d}, {d}, {d}]", .{ @as(f64, @floatFromInt(l.read_call)) / 1e3, @as(f64, @floatFromInt(l.read_end)) / 1e3, @intFromBool(l.missed), l.gate_gu, l.gate_down });
            first = false;
        }
        try w.writeAll("]}");
    }
    try w.writeAll("], \"signal_fields\": [\"value\", \"signal_call\", \"bytes_landed\", \"call_ns\"], \"signals\": [");
    for (signals[0..n_signals], 0..) |sg, i| try w.print("{s}[{d}, {d:.1}, {d:.1}, {d}]", .{ if (i == 0) "" else ", ", sg[0], us(@intCast(sg[1])), us(@intCast(sg[2])), sg[3] });
    try w.writeAll("]}");
}

test "dsv41 verify timeline: a default build stamps nothing and links no shim; the storage bound is static" {
    try std.testing.expectEqual(@as(usize, (1 << 18) * 56 + 1024 * (64 * @sizeOf(Layer) + 16) + (1 << 17) * 32), storage_bytes);
    try std.testing.expectError(error.TimelineCycles, bound(1026, 40));
    try std.testing.expectError(error.TimelineLayers, bound(1024, 65));
    try bound(1024, 40);
    if (enabled) return error.SkipZigTest;
    try std.testing.expect(!@hasDecl(c, "dsv41tl_install"));
    cycleBegin();
    verifyBegin();
    point(0, .call, 0, false);
    gate(0, 1, 2, 1);
    try std.testing.expectEqual(@as(u32, 0), cycles);
    try std.testing.expect(!active);
}

// Profile builds, a device step (DSV41_PHASE0B_MLX=1; no model): MLX's own buffers on the GPU stream reach the hook.
// Prints TIMELINE_HOOK_SMOKE {pass, calls, tagged, rows, classes, names}; fails when no commit was seen or recorded.
test "dsv41 smoke 0b: the command-buffer hook sees MLX's own commits on the GPU stream (no model)" {
    if (comptime !enabled) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const ops = @import("../ops.zig");
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(std.testing.allocator, s);
    defer g.deinit();
    var vals: [1024]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const x = try g.hostArray(std.mem.sliceAsBytes(&vals), &.{1024}, .float32);
    try g.evalAll(&.{try g.add(x, x)});
    try install(s, 2, 1);
    defer uninstall();
    cycleBegin();
    verifyBegin();
    const y = try g.add(try g.mul(x, x), x);
    try g.evalAll(&.{y});
    _ = mlx.mlx_synchronize(s);
    verifyEnd();
    const pending = settle(s, 2000);
    const hs = hookStats();
    const k = counts();
    var names: [512]u8 = undefined;
    const nlen = c.dsv41tl_hooked_names(&names, names.len);
    const pass = hs[0] > 0 and k[0] > 0 and pending == 0;
    std.debug.print("NATIVE TIMELINE_HOOK_SMOKE {{\"pass\": {}, \"calls\": {d}, \"tagged\": {d}, \"other_queue\": {d}, \"rows\": {d}, \"pending\": {d}, \"classes\": {d}, \"names\": \"{s}\", \"first_row_gpu_us\": {d:.1}}}\n", .{ pass, hs[0], hs[1], hs[2], k[0], pending, hs[3], names[0..nlen], if (k[0] > 0) @as(f64, @floatFromInt(rows[0].gpu_end -| rows[0].gpu_start)) / 1e3 else 0 });
    try std.testing.expect(pass);
}

var selftest_rows: [8]Row = undefined;

test "dsv41 verify timeline (profile builds): the -commit hook on a fake buffer hierarchy: a subclass overriding -commit records once, a sibling through its base, an unrelated class after a rehook" {
    if (comptime !enabled) return error.SkipZigTest;
    // dsv41tl_selftest (lib/expert_io/dsv41_cb_timeline.mm): no Metal object; 0 or the failed check's number
    try std.testing.expectEqual(@as(c_int, 0), c.dsv41tl_selftest(&selftest_rows, selftest_rows.len));
    // the Zig side reaches the shim's counters (the fake commits: 5 override entries recorded as rows)
    try std.testing.expect(hookStats()[0] > 0);
    try std.testing.expect(counts()[0] > 0);
}

test "dsv41 verify timeline (profile builds): a hand-made verify: busy union, idle split, hit spans, router gaps, the gated waits to the pool's signals" {
    if (comptime !enabled) return error.SkipZigTest;
    const tg = struct {
        fn of(cyc: u64, p: Phase) u64 {
            return (cyc + 1) << 8 | @intFromEnum(p);
        }
    }.of;
    // Verify [100, 300) of cycle 0. Layer 0: barrier eval commits R0 at 112 (GPU 105..120), route 113, hit wave
    // committed 125 (GPU 124..140), hit stamp 130; gated: gu value 7, last down 9. Layer 1: its barrier eval commits
    // R1 at 155 (GPU 140..200: waits on layer 0's bytes then runs), route 160, hit H1 committed 165 (GPU 205..215),
    // hit stamp 170. A buffer of cycle 1 is ignored. The pool signals value 7 at 150 and 9 at 180.
    const rs = [_]Row{
        .{ .host_commit = 112, .gpu_start = 105, .gpu_end = 120, .tag = tg(0, .verify) },
        .{ .host_commit = 125, .gpu_start = 124, .gpu_end = 140, .tag = tg(0, .verify) },
        .{ .host_commit = 155, .gpu_start = 140, .gpu_end = 200, .tag = tg(0, .verify) },
        .{ .host_commit = 165, .gpu_start = 205, .gpu_end = 215, .tag = tg(0, .verify) },
        .{ .host_commit = 166, .gpu_start = 130, .gpu_end = 140, .tag = tg(1, .verify) },
    };
    var ls: [3]Layer = @splat(.{});
    ls[0] = .{ .t = .{ 110, 114, 115, 130, 131 }, .read_call = 1000, .read_end = 1500, .missed = true, .gate_gu = 7, .gate_down = 9 };
    ls[1] = .{ .t = .{ 150, 156, 160, 170, 171 }, .read_call = 1500, .read_end = 4000 };
    const sig = [_]Signal{ .{ 5, 90, 89, 1 }, .{ 7, 150, 149, 1 }, .{ 9, 180, 178, 1 } };
    var scratch: [8]Span = undefined;
    var per: [3]PerLayer = undefined;
    const s = cycleSum(&rs, .{ .begin = 100, .end = 300 }, &ls, 0, &sig, &scratch, &per);
    // busy: 105..120, 124..200, 205..215 = 15 + 76 + 10 = 101; idle 99
    try std.testing.expectEqual(@as(u64, 200), s.wall);
    try std.testing.expectEqual(@as(u64, 101), s.busy);
    try std.testing.expectEqual(@as(u64, 99), s.idle);
    // barrier idle: layer 0 [110, 130) = 120..124 = 4; layer 1 [150, 170) = 0. encode: [100, 110) = 5, [130, 150) = 0.
    // tail [170, 300) = 170..205 is busy to 200: 200..205 + 215..300 = 90
    try std.testing.expectEqual(@as(u64, 4), s.idle_barrier);
    try std.testing.expectEqual(@as(u64, 5), s.idle_encode);
    try std.testing.expectEqual(@as(u64, 90), s.idle_tail);
    try std.testing.expectEqual(s.idle, s.idle_barrier + s.idle_encode + s.idle_tail);
    // hit spans 124..140 = 16 and 205..215 = 10; router-to-hit gaps 124 - 120 = 4 and 205 - 200 = 5
    try std.testing.expectEqual(@as(?u64, 16), per[0].hit_span);
    try std.testing.expectEqual(@as(?u64, 10), per[1].hit_span);
    try std.testing.expectEqual(@as(u64, 26), s.hit_span);
    try std.testing.expectEqual(@as(?u64, 4), per[0].gap);
    try std.testing.expectEqual(@as(?u64, 5), per[1].gap);
    // layer 0's gated waits: from max(R1 start 140, H0 end 140) = 140 to the last down signal 180 = 40, to gu's 150 = 10
    try std.testing.expectEqual(@as(?u64, 40), per[0].wait);
    try std.testing.expectEqual(@as(?u64, 10), per[0].wait_gu);
    try std.testing.expectEqual(@as(u64, 101 - 40), s.work);
    try std.testing.expectEqual(@as(u32, 1), s.gated);
    try std.testing.expectEqual(@as(u32, 0), s.unsignalled);
    try std.testing.expectEqual(@as(u64, 3000), s.read);
    // layer 0 missed: busy from its hit (130) to layer 1's barrier submit (150) = 130..150
    try std.testing.expectEqual(@as(u64, 20), s.miss_busy);
    try std.testing.expectEqual(@as(u32, 4), s.buffers);
    try std.testing.expectEqual(@as(u32, 2), s.layers);
    // a signal log that stops short leaves the gate unsignalled (no wait counted)
    const s2 = cycleSum(&rs, .{ .begin = 100, .end = 300 }, &ls, 0, sig[0..2], &scratch, &per);
    try std.testing.expectEqual(@as(u32, 1), s2.unsignalled);
    try std.testing.expectEqual(@as(u64, 0), s2.wait);
    // the install refuses without an MLX command buffer (no device touched)
    try std.testing.expectEqual(@as(c_int, -1), c.dsv41tl_install(null, &rows, 1));
}
