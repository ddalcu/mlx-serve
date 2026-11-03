//! The routed experts of the native DeepSeek-V4.1 model over the expert
//! streamer (track M, M3). Oracle: the EXL3 decode lane of the Python tier,
//! `plane_lane.PackedDecode.run` driving `exl3_lane.Exl3PackedOps`.
//!
//! Per routed-layer call the trunk hands the hook `xf [n, hidden]` and the
//! router's `indices [n, k]`. The hook reads the ids on the host (the routing
//! barrier), asks an expert source for their slots (`route`), runs the
//! residents' wave at once, then each miss part's gate/up once its gate/up
//! bytes landed (`waitGu`) and its down once the rest landed (`waitDown`),
//! and hands the call back (`release`: the rows stay pinned until the next
//! route's flush, after the eval that consumed them). The outputs come back
//! in the router's order as `[n, k, hidden]` f32; the trunk combines them.
//! `grow` is the one phase change; `flush` follows the forward's last eval.
//! A call of more rows than a route takes (a prefill chunk) is the wide lane:
//! the math's `prefill` over the slots its experts are served in.
//!
//! Sources: `StreamSource` (the streamer's `Stream`) and `FakeSource` (the
//! streamer's residency policy with no reads, for host tests). The math of a
//! group of rows sharing a bank is the C2 `quant` seam (quant.zig): `QuantMath`
//! over an accepted quant (the EXL3 quant: PREP=rin + the decode GEMV, DIG-X
//! prefill waves) is the served math; `EagerChain` is the stock tier's op chain
//! around the quant's decode GEMV (the parity harnesses); `TraceGemv` /
//! `TraceMath` stand in on the trace backend.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const ops = @import("ops.zig");
const expert_bank = @import("expert/bank.zig");
const expert_io = sdk_ext.expert.io;
const expert_policy = @import("sdk_ext.zig").expert.policy;
const expert_stream = @import("expert/stream.zig");
const expert_lookahead = @import("expert/lookahead.zig");
const expert_event = sdk_ext.expert.event;
const xk = @import("../../quant/exl3/stream/kernels.zig");
const sdk_ext = @import("sdk_ext.zig");
const quant = @import("sdk_ext.zig").quant;
const dt = @import("profile/decode_timers.zig");
const timeline = @import("profile/verify_timeline.zig");
const first_cycle = @import("profile/decode_first.zig");
const prof = @import("profile/prefill_timers.zig");
const xq = @import("../../quant/exl3/stream/quant.zig");

pub const max_route_ids = expert_policy.max_route_ids;
/// The widest decode / verify forward (the RC routes' and the verify block's rows): at most
/// `decode_forward_rows * top_k` routed ids, which construction proves fit one route (`max_route_ids`),
/// so a forward of <= 8 rows never reaches the wide lane.
pub const decode_forward_rows: u32 = 8;
/// The fewest routed ids a wide-lane call takes (one more than a route): refused below, by name.
pub const wide_min_ids: u32 = max_route_ids + 1;
pub const BankKind = expert_stream.BankKind;
pub const SlotRef = expert_stream.SlotRef;
pub const Stats = expert_stream.Stats;
pub const Error = expert_stream.Error;
const Load = expert_policy.Load;
const n_banks = std.meta.fieldNames(BankKind).len;

/// What one routed-layer call serves (`sdk_ext.expert.Served`).
pub const Served = sdk_ext.expert.Served;

/// One projection's slot arrays in a bank (the streamer's `ProjArrays` over
/// any backend): code int16 [rows, in/16, out/16, 16K], rout f16 [rows, out],
/// rin f16 [rows, in].
pub const ProjOf = xq.ProjArrays;

/// A bank's nine arrays by projection (the streamer's `BankArrays`; the quant seam's).
pub fn BankArraysOf(comptime T: type) type {
    return quant.BankArrays(ProjOf(T));
}

// ── The source contract (`sdk_ext.expert.assertSource`: the required set and each declared capability's) ──

pub const assertSource = sdk_ext.expert.assertSource;

/// The stream's wave numbering of a call's ids (`expert_stream.wavesOf`).
const wavesOf = expert_stream.wavesOf;

/// Trace arrays in one record's geometry (`expert_stream.traceBank`).
const traceBank = expert_stream.traceBank;

/// DEVROUTE's resident map of one layer from its policy (`expert_stream.lutOf`).
const lutOf = expert_stream.lutOf;

// ── StreamSource: the streamer's Stream, the EXL3 source's side of the contract (`expert_stream.StreamSource`) ──

pub const StreamSource = expert_stream.StreamSource;

// ── FakeSource: the streamer's residency with no reads (host tests) ──

/// The streamer's residency policy (one `LayerPolicy` per layer, the Stream's
/// part cutting and slot arithmetic) with every load landed at once and no
/// memory: the fixture-backed stand-in for host tests. It keeps the Stream's
/// call rules (a route flushes released calls first; four calls at most;
/// `grow` once, with every call released) and logs every call it gets.
pub const FakeSource = struct {
    a: std.mem.Allocator,
    geom: expert_bank.Layer,
    n_experts: u32,
    policies: []expert_policy.LayerPolicy,
    base_rows: []u32,
    ext_rows: []u32,
    transient_rows: u32,
    records_per_part: u32,
    phase: expert_policy.Phase = .prefill,
    calls: [4]Call = @splat(.{}),
    log: std.ArrayList(Event) = .empty,
    /// Slots a deferred call still reads (`holdBase`): held in every later plan of `held_layer`.
    held: std.ArrayList(u32) = .empty,
    held_layer: u32 = 0,
    counters: Stats = .{},
    /// When set, events are stamped with its node count (op order checks).
    trace: ?*const ops.TraceOps = null,
    /// The lookahead's selection on the next layer's gate scores (the Stream's
    /// `speculate`), logged per call that passes scores.
    selector: ?expert_lookahead.Selector = null,
    picks: std.ArrayList(Pick) = .empty,
    gate_value: u64 = 0,
    /// P1: the layer whose read-ahead is live (the Stream's rule: its first route or the barrier lands it),
    /// and every admission in order (the tests' view of the seed the model handed over).
    ahead_layer: ?u32 = null,
    ahead_log: std.ArrayList(u16) = .empty,

    pub const caps: sdk_ext.expert.Caps = .{ .two_phase = true, .prompt_seed = true, .read_ahead = true, .wide = true, .event_gates = true };

    pub const Pick = struct { layer: u32, n: u8 = 0, experts: [expert_lookahead.max_budget]u16 = undefined };

    pub const Call = struct {
        state: enum { free, live, released } = .free,
        layer: u32 = 0,
        plan: expert_policy.Plan = .{},
        n_parts: u32 = 0,
        refs: [max_route_ids]SlotRef = undefined,
        waves: [max_route_ids]u8 = undefined,
    };

    pub const Event = struct {
        pub const Kind = enum { route, wait_gu, wait_down, release, flush, grow, gate, read_ahead, await_read_ahead };
        kind: Kind,
        layer: u32 = 0,
        part: u32 = 0,
        /// Trace nodes recorded before the event (0 without `trace`).
        at: usize = 0,
    };

    pub const Options = struct {
        hidden: u64,
        inter: u64,
        n_experts: u32,
        /// Persistent rows per layer before `grow`.
        rows: []const u32,
        transient_rows: u32 = max_route_ids,
        records_per_part: u32 = 3,
    };

    pub fn init(a: std.mem.Allocator, opt: Options) !FakeSource {
        const geom = expert_bank.layerSegments(3, opt.hidden, opt.inter) orelse return error.InvalidGeometry;
        const n = opt.rows.len;
        const policies = try a.alloc(expert_policy.LayerPolicy, n);
        errdefer a.free(policies);
        var n_init: usize = 0;
        errdefer for (policies[0..n_init]) |*p| p.deinit(a);
        for (policies, opt.rows) |*p, rows| {
            p.* = try expert_policy.LayerPolicy.init(a, opt.n_experts, rows);
            n_init += 1;
        }
        const base = try a.dupe(u32, opt.rows);
        errdefer a.free(base);
        const ext = try a.alloc(u32, n);
        @memset(ext, 0);
        return .{
            .a = a,
            .geom = geom,
            .n_experts = opt.n_experts,
            .policies = policies,
            .base_rows = base,
            .ext_rows = ext,
            .transient_rows = opt.transient_rows,
            .records_per_part = opt.records_per_part,
        };
    }

    pub fn deinit(self: *FakeSource) void {
        if (self.selector) |*sel| sel.deinit(self.a);
        self.picks.deinit(self.a);
        for (self.policies) |*p| p.deinit(self.a);
        self.a.free(self.policies);
        self.a.free(self.base_rows);
        self.a.free(self.ext_rows);
        self.log.deinit(self.a);
        self.held.deinit(self.a);
        self.ahead_log.deinit(self.a);
        self.* = undefined;
    }

    fn note(self: *FakeSource, ev: Event) void {
        var e = ev;
        if (self.trace) |t| e.at = t.nodes.items.len;
        self.log.append(self.a, e) catch @panic("fake source log: out of memory");
    }

    pub fn holdBase(self: *FakeSource, call: *Call) !void {
        if (self.held.items.len > 0 and self.held_layer != call.layer) return error.HeldOtherLayer;
        self.held_layer = call.layer;
        const pol = &self.policies[call.layer];
        for (call.plan.hitsOf()) |e| {
            const s = pol.slotOf(e).?;
            if (s < pol.capacity) try self.held.append(self.a, s);
        }
        for (call.plan.loadsOf()) |l| if (l.persistent) try self.held.append(self.a, l.slot);
    }

    pub fn releaseHeld(self: *FakeSource) void {
        self.held.clearRetainingCapacity();
    }

    pub fn seedPrefill(self: *FakeSource, layer: u32, ids: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        self.policies[layer].prepareSeed(ids);
    }

    pub fn seedRanks(self: *const FakeSource, layer: u32) u32 {
        return self.policies[layer].seed_ranks;
    }

    /// P1: the Stream's admission (`LayerPolicy.admitReadAhead`), no reads; `part` = experts admitted.
    pub fn readAheadSeed(self: *FakeSource, layer: u32, experts: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        if (self.ahead_layer) |l| try self.awaitReadAhead(l);
        var out: [max_ahead]expert_policy.LayerPolicy.ReadAhead = undefined;
        const admitted = self.policies[layer].admitReadAhead(experts, &out);
        const n = admitted.len;
        for (admitted) |r| self.ahead_log.append(self.a, r.expert) catch @panic("fake source read-ahead log: out of memory");
        if (n > 0) self.ahead_layer = layer;
        self.note(.{ .kind = .read_ahead, .layer = layer, .part = @intCast(n) });
    }

    pub fn awaitReadAhead(self: *FakeSource, layer: u32) Error!void {
        if (self.ahead_layer != layer) return;
        self.ahead_layer = null;
        self.note(.{ .kind = .await_read_ahead, .layer = layer });
    }

    const max_ahead = 1024;

    pub fn slotRef(self: *const FakeSource, layer: u32, slot: u32) SlotRef {
        const cap = self.policies[layer].capacity;
        if (slot < self.base_rows[layer]) return .{ .bank = .base, .row = slot };
        if (slot < cap) return .{ .bank = .ext, .row = slot - self.base_rows[layer] };
        return .{ .bank = .transient, .row = slot - cap };
    }

    pub fn route(self: *FakeSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
        std.debug.assert(ids.len > 0 and ids.len <= max_route_ids);
        std.debug.assert(scores.len == 0 or self.selector != null);
        if (self.ahead_layer == layer) try self.awaitReadAhead(layer);
        try self.flush();
        const call = for (&self.calls) |*c| {
            if (c.state == .free) break c;
        } else return error.RoutesExhausted;
        call.* = .{ .layer = layer };
        const plan = &call.plan;
        const held: []const u32 = if (self.held.items.len > 0 and self.held_layer == layer) self.held.items else &.{};
        self.policies[layer].planWith(ids, self.phase, plan, .{ .held = held });
        // A row a live call still serves from, or a deferred call still reads, is never refilled.
        for (plan.loadsOf()) |l| {
            if (std.mem.indexOfScalar(u32, held, l.slot) != null) return error.SlotStillPinned;
            for (&self.calls) |*o| {
                if (o == call or o.state != .live or o.layer != layer) continue;
                if (std.mem.indexOfScalar(u32, o.plan.slotsOf(), l.slot) != null) return error.SlotStillPinned;
            }
        }
        // The Stream's parts: loads in file (expert) order; decode = bounded
        // parts of `records_per_part` (no two records touch), prefill = pool-job chunks.
        var order: [max_route_ids]Load = undefined;
        const n = plan.n_loads;
        @memcpy(order[0..n], plan.loadsOf());
        std.sort.insertion(Load, order[0..n], {}, struct {
            fn less(_: void, x: Load, y: Load) bool {
                return x.expert < y.expert;
            }
        }.less);
        var offsets: [max_route_ids]u64 = undefined;
        var lengths: [max_route_ids]u64 = undefined;
        for (order[0..n], 0..) |l, i| {
            offsets[i] = @as(u64, l.expert) * self.geom.record_bytes;
            lengths[i] = self.geom.logical_bytes;
        }
        var ends_buf: [max_route_ids]u32 = undefined;
        const ends = if (self.phase == .decode)
            expert_policy.boundedParts(offsets[0..n], lengths[0..n], self.records_per_part, &ends_buf)
        else blk: {
            var k: u32 = 0;
            var e: u32 = 0;
            while (e < n) : (k += 1) {
                e = @min(e + expert_io.max_items, n);
                ends_buf[k] = e;
            }
            break :blk ends_buf[0..k];
        };
        var parts: [max_route_ids][]const Load = undefined;
        var start: u32 = 0;
        for (ends, 0..) |end, p| {
            parts[p] = order[start..end];
            start = end;
        }
        call.n_parts = @intCast(ends.len);
        var hit_slots: [max_route_ids]u32 = undefined;
        for (plan.hitsOf(), hit_slots[0..plan.n_hits]) |e, *s| s.* = self.policies[layer].slotOf(e).?;
        wavesOf(plan, hit_slots[0..plan.n_hits], parts[0..ends.len], call.waves[0..ids.len]);
        for (plan.slotsOf(), call.refs[0..ids.len]) |s, *r| r.* = self.slotRef(layer, s);
        const c = &self.counters;
        c.route_calls += 1;
        c.expert_cache_hits += plan.n_hits;
        c.expert_cache_misses += plan.n_misses;
        c.expert_cache_evictions += plan.n_evictions;
        for (plan.loadsOf()) |l| {
            if (l.persistent) c.persistent_loads += 1 else c.transient_loads += 1;
        }
        if (scores.len > 0 and self.phase == .decode and layer + 1 < self.policies.len) {
            const sel = &self.selector.?;
            var pick: Pick = .{ .layer = layer + 1 };
            pick.n = @intCast(sel.select(scores, &self.policies[layer + 1], pick.experts[0..sel.budget]).len);
            self.picks.append(self.a, pick) catch @panic("fake source picks: out of memory");
        }
        call.state = .live;
        self.note(.{ .kind = .route, .layer = layer });
        return call;
    }

    /// The Stream's gate values: the call's gate/up wave at `gu`, part p's down at `down_first + p`.
    pub fn gate(self: *FakeSource, call: *Call) Error!?expert_stream.Gates {
        std.debug.assert(call.state == .live);
        if (call.n_parts == 0) return null;
        const lo = self.gate_value;
        self.gate_value = lo + 1 + call.n_parts;
        self.note(.{ .kind = .gate, .layer = call.layer, .part = call.n_parts });
        return .{ .gu = lo + 1, .down_first = lo + 2, .n_parts = call.n_parts };
    }

    /// DEVROUTE: as `StreamSource.residentLut`.
    pub fn residentLut(self: *const FakeSource, layer: u32, out: []u32) void {
        lutOf(&self.policies[layer], self, layer, out);
    }

    pub fn decoding(self: *const FakeSource) bool {
        return self.phase == .decode;
    }

    pub fn served(_: *FakeSource, call: *const Call) Served {
        const n = call.plan.n_ids;
        return .{ .refs = call.refs[0..n], .waves = call.waves[0..n], .n_parts = call.n_parts };
    }

    pub fn waitGu(self: *FakeSource, call: *Call, part: u32) Error!void {
        std.debug.assert(call.state == .live and part < call.n_parts);
        self.note(.{ .kind = .wait_gu, .layer = call.layer, .part = part });
    }

    pub fn waitDown(self: *FakeSource, call: *Call, part: u32) Error!void {
        std.debug.assert(call.state == .live and part < call.n_parts);
        self.note(.{ .kind = .wait_down, .layer = call.layer, .part = part });
    }

    pub fn release(self: *FakeSource, call: *Call) void {
        std.debug.assert(call.state == .live);
        call.state = .released;
        self.note(.{ .kind = .release, .layer = call.layer });
    }

    pub fn flush(self: *FakeSource) Error!void {
        var any = false;
        for (&self.calls) |*c| if (c.state == .released) {
            c.state = .free;
            any = true;
        };
        if (any) self.note(.{ .kind = .flush });
    }

    pub fn grow(self: *FakeSource, decode_rows: []const u32) !void {
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (decode_rows.len != self.policies.len) return error.InvalidRows;
        try self.flush();
        for (&self.calls) |*c| if (c.state != .free) return error.RoutesLive;
        for (self.policies, decode_rows) |*p, rows| {
            if (rows < p.capacity or rows > self.n_experts) return error.InvalidRows;
        }
        for (self.policies, decode_rows, self.ext_rows) |*p, rows, *ext| {
            ext.* = rows - p.capacity;
            p.grow(rows) catch unreachable;
        }
        self.phase = .decode;
        self.note(.{ .kind = .grow });
    }

    pub fn stats(self: *FakeSource) Stats {
        return self.counters;
    }

    pub fn isResident(self: *const FakeSource, layer: u32, expert: u16) bool {
        return self.policies[layer].slotOf(expert) != null;
    }

    pub fn bankRows(self: *FakeSource, layer: u32, kind: BankKind) u32 {
        return switch (kind) {
            .base => self.base_rows[layer],
            .ext => self.ext_rows[layer],
            .transient => self.transient_rows,
        };
    }

    pub fn bankArrays(self: *FakeSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
        if (@TypeOf(g.*) != ops.TraceOps) @compileError("FakeSource binds trace arrays only");
        const rows = self.bankRows(layer, kind);
        if (rows == 0) return null;
        return try traceBank(g, &self.geom, rows);
    }

    /// Calls that are neither free nor released (tests).
    pub fn liveCalls(self: *const FakeSource) usize {
        var n: usize = 0;
        for (self.calls) |c| n += @intFromBool(c.state == .live);
        return n;
    }
};

// ── The math seam ──

/// `128 ** -0.5` as the Python float rounds to the f32 MLX takes.
const t128_scale: f32 = @bitCast(@as(u32, 0x3db504f3));

/// The stock tier's routed-expert math, `exl3_lane.Exl3PackedOps` op for op:
/// per projection `t128(gemv(t128(x * rin[slot])) ) * rout[slot]` (f32), the
/// clamped SwiGLU between gate/up and down. `Gemv.project(g, proj, xh, ids,
/// code)` is the EXL3 decode GEMV (`z = xh @ W_hat[slot]` in the trellis domain,
/// f32 [rows, out]): the accepted EXL3 quant's (`*const exl3_quant.Gemv(G)`).
/// The parity harnesses' math (the stock path reads f32 routed rows).
pub fn EagerChain(comptime G: type, comptime Gemv: type) type {
    return struct {
        const Self = @This();
        const T = G.T;

        gemv: Gemv,
        swiglu_limit: f64,

        pub fn init(gemv: Gemv, c: anytype) Self {
            return .{ .gemv = gemv, .swiglu_limit = c.swiglu_limit };
        }

        /// `tcq_runtime._t128`: the normalised 128-block Walsh-Hadamard over the last axis.
        fn t128(g: *G, v: T) !T {
            const sh = g.shapeOf(v);
            var blocks = sh;
            blocks.d[sh.n - 1] = @divExact(sh.d[sh.n - 1], 128);
            blocks.d[sh.n] = 128;
            blocks.n += 1;
            const h = try g.hadamard(try g.reshape(try g.astype(v, .float32), blocks.slice()), t128_scale);
            return g.reshape(h, sh.slice());
        }

        /// `Exl3PackedOps._project`.
        fn project(self: *const Self, g: *G, x: T, ids: T, p: ProjOf(T), proj: xq.Proj) !T {
            const rin = try g.astype(try g.take(p.rin, ids, 0), .float32);
            const xh = try t128(g, try g.mul(try g.astype(x, .float32), rin));
            const z = try self.gemv.project(g, proj, xh, ids, p.code);
            const rout = try g.astype(try g.take(p.rout, ids, 0), .float32);
            return g.mul(try t128(g, z), rout);
        }

        /// Gate and up of rows `x` [rows, hidden] at slot rows `ids` [rows]
        /// (uint32), then `_clamped_swiglu(g, u, limit)`: f32 [rows, inter].
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const gg = try self.project(g, x, ids, gate, .gate);
            const uu = try self.project(g, x, ids, up, .up);
            const lim = self.swiglu_limit;
            const uc = try g.clip(uu, try g.scalar(-lim, .float32), try g.scalar(lim, .float32));
            const gc = try g.minimum(gg, try g.scalar(lim, .float32));
            return g.mul(try g.silu(gc), uc);
        }

        /// Down of the SwiGLU rows `h` [rows, inter]: f32 [rows, hidden].
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.project(g, h, ids, d, .down);
        }
    };
}

/// The served routed-expert math: an accepted C2 quant (`quant.checkAccepted`),
/// its decode `gateUp` / `down` (the EXL3 quant: PREP=rin around the decode
/// GEMV) and its prefill waves. The quant reads bf16 routed rows: a call whose
/// rows arrive in another dtype is rounded to bf16 once, here (the RC tier's
/// MoE input is bf16 already).
pub fn QuantMath(comptime G: type, comptime Q: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        /// The widest decode-lane call (`gateUp` / `down`).
        pub const max_decode_rows: u32 = Q.max_decode_rows;
        q: *Q,

        pub fn init(q: *Q, _: anytype) Self {
            return .{ .q = q };
        }

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            return self.q.gateUp(g, xb, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.q.down(g, h, ids, d);
        }

        /// The quant's banked route, when it has one (a construction-time type choice; `Options.banked` binds it).
        pub const has_banked = @hasDecl(Q, "gateUpBanked");

        /// ids packed (bank << 24 | slot row); `banks` base, ext, transient.
        pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [n_banks]BankArraysOf(T)) !T {
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            return self.q.gateUpBanked(g, xb, ids, banks);
        }

        pub fn downBanked(self: *const Self, g: *G, h: T, ids: T, banks: *const [n_banks]BankArraysOf(T)) !T {
            return self.q.downBanked(g, h, ids, banks);
        }

        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T)) !T {
            return self.q.prefill(g, layer, x, rows, bank);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            return self.q.finishPrefill(g);
        }

        /// The quant's unjoined prefill, when it has one (a construction-time type choice).
        pub const has_parts = @hasDecl(Q, "prefillParts");

        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T), alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            return self.q.prefillParts(g, layer, x, rows, bank, alloc, outs, pos);
        }
    };
}

/// A decode math `D` with one prefill route `P` per layer (`call(g, x, rows,
/// bank)` / `finish(g)`: the quant's prefill shape), for host tests.
pub fn WithPrefillRoutes(comptime G: type, comptime D: type, comptime P: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        pub const max_decode_rows: u32 = if (@hasDecl(D, "max_decode_rows")) D.max_decode_rows else max_route_ids;
        d: D,
        routes: []P,

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            return self.d.gateUp(g, x, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.d.down(g, h, ids, d);
        }

        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T)) !T {
            return self.routes[layer].call(g, x, rows, bank);
        }

        pub const has_parts = @hasDecl(P, "callParts");

        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T), alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            return self.routes[layer].callParts(g, x, rows, bank, alloc, outs, pos);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            for (self.routes) |*r| try r.finish(g);
        }
    };
}

/// The GEMV's launch contract on the trace backend (the kernel manifest's
/// inputs: xh f32 [rows, in], ids uint32 [rows], code int16 [cap, in/16,
/// out/16, 48] at K 3); the output is a kernel node f32 [rows, out].
pub const TraceGemv = struct {
    pub fn project(_: TraceGemv, g: *ops.TraceOps, _: xq.Proj, xh: u32, ids: u32, code: u32) !u32 {
        const sx = g.shapeOf(xh);
        const si = g.shapeOf(ids);
        const sc = g.shapeOf(code);
        if (g.dtypeOf(xh) != .float32 or g.dtypeOf(ids) != .uint32 or g.dtypeOf(code) != .int16) return error.GemvDtype;
        if (sx.n != 2 or si.n != 1 or sc.n != 4 or si.d[0] != sx.d[0]) return error.GemvShape;
        if (sc.d[1] * 16 != sx.d[1] or sc.d[3] != 48) return error.GemvShape;
        return g.kernel(&.{ sx.d[0], sc.d[2] * 16 }, .float32);
    }
};

/// Shape-level math for trace models whose widths no EXL3 bank has (the mini
/// config's hidden 64): one kernel node per projection pair.
pub const TraceMath = struct {
    hidden: c_int,
    inter: c_int,

    pub fn gateUp(self: *const TraceMath, g: *ops.TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        _ = gate;
        _ = up;
        if (g.shapeOf(x).d[1] != self.hidden or g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .float32);
    }

    pub fn down(self: *const TraceMath, g: *ops.TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        _ = d;
        if (g.shapeOf(h).d[1] != self.inter or g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .float32);
    }
};

// ── The executor: the model's routed hook over a source ──

/// `argsort(positions)` of unique positions: `inv[pos[j]] = j` (the row of
/// the joined outputs that holds routed position `pos[j]`).
fn invertPositions(pos: []const u32, inv: []u32) void {
    for (pos, 0..) |p, j| inv[p] = @intCast(j);
}

/// The most sources JOINLESS's combine reads (`dsv41_kernel_routes.JoinlessCombine.sources`).
pub const joinless_sources = 24;

/// JOINLESS's sources for outputs of `rows` rows each (join order), the minimal copy: up to
/// `joinless_sources` outputs are each read in place; above, the smallest `n - 23` (earlier first among
/// equals) are concatenated in join order into the last source and the rest are sources 0.. in join order.
/// Per output: `src[i]` its source, `off[i]` its first row there; returns the sources. `order`: scratch.
/// Exact: a concatenate copies words, and the combine folds each token's k products in k order.
pub fn planJoinless(rows: []const u32, order: []u32, src: []u8, off: []u32) usize {
    return planJoinlessBound(rows, order, src, off, joinless_sources);
}

/// `planJoinless` with the combine's own source bound: up to `bound` outputs each read in place (`bound` >= the
/// outputs: no merge); above it the stock rule at `joinless_sources`. A combine that reads every source in place (MiMo
/// P7 `prefill_combine: "inplace"`) passes its bound; every other caller keeps `planJoinless`.
pub fn planJoinlessBound(rows: []const u32, order: []u32, src: []u8, off: []u32, bound: usize) usize {
    const n = rows.len;
    @memset(off, 0);
    if (n <= @max(bound, joinless_sources)) {
        for (src, 0..) |*s, i| s.* = @intCast(i);
        return n;
    }
    for (order, 0..) |*x, i| x.* = @intCast(i);
    std.sort.pdq(u32, order, rows, struct {
        fn lt(r: []const u32, x: u32, y: u32) bool {
            return if (r[x] != r[y]) r[x] < r[y] else x < y;
        }
    }.lt);
    const merged: u8 = joinless_sources - 1;
    @memset(src, 0);
    for (order[0 .. n - merged]) |i| src[i] = merged;
    var in_place: u8 = 0;
    var at: u32 = 0;
    for (rows, src, off) |r, *s, *o| {
        if (s.* == merged) {
            o.* = at;
            at += r;
        } else {
            s.* = in_place;
            in_place += 1;
        }
    }
    return joinless_sources;
}

/// Each assignment's (source, row) under a `planJoinless` plan, as int32 pairs: output i's join-ordered
/// rows (`pos`, the outputs' rows back to back) are rows `off[i]..` of source `src[i]`.
pub fn fillJoinlessLoc(rows: []const u32, src: []const u8, off: []const u32, pos: []const u32, loc: []i32) void {
    var j: usize = 0;
    for (rows, src, off) |r, s, o| {
        for (pos[j..][0..r], o..) |p, row| {
            loc[2 * p] = s;
            loc[2 * p + 1] = @intCast(row);
        }
        j += r;
    }
}

/// The merge's host scratch, reused across calls.
pub fn JoinlessScratch(comptime T: type) type {
    return struct {
        rows: std.ArrayList(u32) = .empty,
        order: std.ArrayList(u32) = .empty,
        src: std.ArrayList(u8) = .empty,
        off: std.ArrayList(u32) = .empty,
        merge_in: std.ArrayList(T) = .empty,

        pub fn deinit(s: *@This(), a: std.mem.Allocator) void {
            inline for (.{ &s.rows, &s.order, &s.src, &s.off, &s.merge_in }) |l| l.deinit(a);
        }
    };
}

/// (profile builds) A wide call's JOINLESS merge for the prefill profile's group line: its outputs, the combine's sources and
/// the rows the one concatenate copied (0: no merge, every output read in place).
pub const MergeStats = struct { outputs: u32 = 0, sources: u32 = 0, copied_rows: u64 = 0 };

/// JOINLESS's merge: `outs` (the KEPT outputs, join order; `pos`: each join-ordered row's assignment) becomes
/// the combine's sources under `planJoinless` (the in-place outputs, then the merged one, KEPT; its inputs
/// released) and `loc` each assignment's (source, row). Returns the rows copied.
pub fn mergeJoinless(comptime G: type, g: *G, a: std.mem.Allocator, outs: *std.ArrayList(G.T), pos: []const u32, loc: []i32, sc: *JoinlessScratch(G.T)) !u64 {
    return mergeJoinlessBound(G, g, a, outs, pos, loc, sc, joinless_sources);
}

/// `mergeJoinless` under the combine's source bound (`planJoinlessBound`).
pub fn mergeJoinlessBound(comptime G: type, g: *G, a: std.mem.Allocator, outs: *std.ArrayList(G.T), pos: []const u32, loc: []i32, sc: *JoinlessScratch(G.T), bound: usize) !u64 {
    const n_out = outs.items.len;
    try sc.rows.resize(a, n_out);
    try sc.order.resize(a, n_out);
    try sc.src.resize(a, n_out);
    try sc.off.resize(a, n_out);
    for (outs.items, sc.rows.items) |x, *r| r.* = @intCast(g.shapeOf(x).dim(0));
    const n_src = planJoinlessBound(sc.rows.items, sc.order.items, sc.src.items, sc.off.items, bound);
    fillJoinlessLoc(sc.rows.items, sc.src.items, sc.off.items, pos, loc);
    if (n_src == n_out) return 0;
    const last: u8 = @intCast(n_src - 1);
    sc.merge_in.clearRetainingCapacity();
    var copied: u64 = 0;
    for (outs.items, sc.src.items, sc.rows.items) |x, s, r| if (s == last) {
        try sc.merge_in.append(a, x);
        copied += r;
    };
    const cat = g.keep(try g.concat(sc.merge_in.items, 0));
    for (sc.merge_in.items) |x| g.release(x);
    var d: usize = 0;
    for (outs.items, sc.src.items) |x, s| if (s != last) {
        outs.items[d] = x;
        d += 1;
    };
    outs.shrinkRetainingCapacity(d);
    outs.appendAssumeCapacity(cat);
    return copied;
}

/// Construction-time routes of the executor.
pub const Routes = struct {
    /// Pass `route` the next routed layer's gate scores (the streamer's
    /// lookahead predictor, evaluated with the routing barrier).
    lookahead: bool = false,
    /// Event gates instead of host waits: every wave is built at once over
    /// event-wait aliases of the bank arrays (the typical tier's gate).
    gated: bool = false,
    /// The wide lane (calls of more than `max_route_ids` routed rows) through
    /// the math's `prefill(g, layer, x, rows, bank)` (a KEPT result) and
    /// `finishPrefill(g)`; off refuses a wide call (`PrefillLaneNotPorted`).
    prefill: bool = false,
};

/// The wide lane's read schedule, chosen at construction (`Options.wide`; both
/// exact: the same rows, slots and kernels, only the order and timing of reads
/// and evals change).
pub const Wide = struct {
    /// Each call seeds its layer's residency with its own ids (the most routed
    /// experts protected in the persistent rows) and drains each group once
    /// (its banks' waves queued).
    seed: bool = false,
    /// Each call feeds its groups hottest first (routed rows descending, ties by
    /// id) instead of in first appearance. `seed` + `hot_first` = the wide feed.
    hot_first: bool = false,
    /// Groups in flight: 2 routes (reads) the next group before this
    /// group's waves (the source's `wideDepth` must allow it).
    depth: u8 = 1,
    /// Experts with at most this many rows in the call run the decode lane's
    /// math (the GEMV, in chunks of the math's `max_decode_rows`) after their
    /// bank's wide waves, not the wide route (0: none). ROUNDING-CLASS against
    /// the wide route: its own reference family.
    cold_rows: u8 = 0,
    /// Every group's base-bank rows of a call built as ONE call's waves after the groups' transient
    /// waves (their slots held until then): fewer, fuller waves. Exact: a wave's rows are
    /// independent of its composition and the combine folds by routed position.
    defer_base: bool = false,
    /// P1: each layer's predicted seed read into its persistent rows (the empty ones, then those of the
    /// residents the prediction leaves out) while its attention runs (the model's predictor pass hands it
    /// over, `Hook.readAheadSeed`, in every prompt pass), landed at the routing barrier before the seed.
    /// Exact: it chooses reads and rows only. Needs `seed`.
    read_ahead: bool = false,
    /// P1b: with `defer_base` and the seed, the deferred base call runs once the seed's groups are read (after the first
    /// group with a transient row), so the later groups' reads overlap its drain; later base rows run in a last call.
    /// Exact: the same rows, slots and kernels; only the call's place in the schedule changes.
    base_at_seed: bool = false,
    /// With `base_at_seed`: the seed's ranks grouped apart from the stream's (the last seed group holds no stream
    /// record) and the base call issued after the last seed group, so only the seed's own reads precede it. Exact:
    /// each expert's rows and kernel are its own whatever its group; the combine folds by routed position.
    seed_aligned: bool = false,
    /// The route-plan cut (glm5next/route-plan, cd6f1ec): the call's rows bucketed by group once (a stable counting sort
    /// into the join's scratch, `bucketRowsByGroup`), each group's bank passes over its own rows only, in the same order
    /// (byte-identical plans), and the distinct experts with their counts in one pass. Off (the control): every bank
    /// pass of every group walks the call's every row. Exact: host planning only.
    route_cut: bool = false,
    pub const max_cold_rows = 8;
};

/// A wide call's rows bucketed by group, each group's rows in ascending row order (a stable counting sort): `ids` the
/// call's routed ids (rows x top-k), `distinct` its experts, `grp_of[e]` expert e's group, `count[e]` (optional) its rows;
/// `off` [n_groups + 1] gets the groups' offsets, `rows` [ids.len] the rows. A pass over `rows[off[g]..off[g + 1]]` meets
/// exactly the rows of group g a pass over the whole call meets, in the same order (exact).
pub fn bucketRowsByGroup(ids: []const u16, distinct: []const u16, grp_of: []const u16, count: ?[]const u32, off: []u32, rows: []u32) void {
    @memset(off, 0);
    if (count) |c| {
        for (distinct) |e| off[grp_of[e] + 1] += c[e];
    } else for (ids) |e| {
        off[grp_of[e] + 1] += 1;
    }
    for (1..off.len) |gi| off[gi] += off[gi - 1];
    // Each row at its group's cursor (`off[gi]` advances to the group's end), then the starts restored.
    for (ids, 0..) |e, row| {
        const gi = grp_of[e];
        rows[off[gi]] = @intCast(row);
        off[gi] += 1;
    }
    var gi = off.len - 1;
    while (gi > 0) : (gi -= 1) off[gi] = off[gi - 1];
    off[0] = 0;
}

/// The routed-expert hook of `Model(G)` over source `S` with math `M`
/// (`gateUp(g, x, ids, gate, up)`, `down(g, h, ids, d)`). Bank arrays are
/// bound once at `init` (base, transient) and at `grow` (the grown rows).
/// `at(layer)` is the per-layer hook the trunk calls.
/// The MLX event wait (the gated arm's): aliases of `xs` into `outs`, which may hold anything (a caller's
/// `undefined`): the shim assigns each alias into its handle, so every handle is made fresh first.
fn mlxEventWait(g: *ops.MlxOps, event: expert_event.Event, xs: []const mlx.mlx_array, value: u64, deps: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    for (outs) |*o| o.* = mlx.mlx_array_new();
    try expert_event.wait(xs, event, value, deps, false, g.s, outs);
    for (outs) |*o| o.* = try g.adopt(o.*);
}

/// A math with optional projection arrays supplies its proven event projection at construction.
/// DeepSeek's default projection always aliases its three EXL3 arrays in their existing order.
fn Exl3EventProjection(comptime T: type) type {
    return struct {
        pub const count: usize = 3;
        pub fn inputs(p: ProjOf(T)) [count]T {
            return .{ p.code, p.rout, p.rin };
        }
        pub fn fromWaited(out: [count]T) ProjOf(T) {
            return .{ .code = out[0], .rout = out[1], .rin = out[2] };
        }
    };
}

pub fn Experts(comptime G: type, comptime S: type, comptime M: type) type {
    return ExpertsWith(G, S, M, .{});
}

pub fn ExpertsWith(comptime G: type, comptime S: type, comptime M: type, comptime routes: Routes) type {
    const Projection = if (@hasDecl(M, "Projection")) M.Projection else ProjOf;
    const EventProjection = if (@hasDecl(M, "EventProjection")) M.EventProjection(G.T) else Exl3EventProjection(G.T);
    const wide_event_gates = routes.gated and (if (@hasDecl(M, "wide_event_gates")) M.wide_event_gates else false);
    const predictor_decode_only = if (@hasDecl(M, "predictor_decode_only")) M.predictor_decode_only else false;
    const wide_profile = if (@hasDecl(M, "Profile")) M.Profile else prof;
    const bound_decode_rows = if (@hasDecl(M, "decode_forward_rows")) M.decode_forward_rows else decode_forward_rows;
    comptime {
        assertSource(S);
        if (routes.gated and !sdk_ext.expert.capsOf(S).event_gates) @compileError(@typeName(S) ++ ": a gated route needs a source with event gates");
        // A source's slot arrays are its quant's: a source and a quant that disagree on the format do not compile.
        if (@hasDecl(S, "Arrays") and S.Arrays(G.T) != Projection(G.T)) @compileError(@typeName(S) ++ ": its slot arrays are not the quant's");
    }
    return struct {
        const Self = @This();
        const T = G.T;
        pub const Arrays = quant.BankArrays(Projection(T));
        /// The math this hook runs (its construction-time capabilities: `has_parts`).
        pub const Math = M;

        /// A routed layer's gate (the lookahead predictor reads the next layer's).
        pub const Gate = struct { w: T, bias: T };

        a: std.mem.Allocator,
        source: *S,
        math: M,
        hidden: c_int,
        n_experts: u32,
        /// Per layer, per bank kind: what the math binds (null: no rows).
        banks: [][n_banks]?Arrays,
        /// Lookahead: every routed layer's gate, by layer.
        gates: []const Gate = &.{},
        /// Gated: the event the stream signals (an MLX backend's MTLSharedEvent).
        event: expert_event.Event = .{ .id = 0, .object = 0 },
        /// The backend the banks were bound on.
        g: *G,
        wide: WideScratch = .{},
        /// The wide lane's read schedule (`Options.wide`).
        wide_route: Wide = .{},
        /// The parts' combine's source bound (`runWideParts`: outputs above it merge into one source); `joinless_sources`
        /// unless the arch installs a combine that reads every source in place (`setJoinlessBound`).
        joinless_bound: usize = joinless_sources,
        /// `releaseTransient` nulled the transient bindings; `grow` binds decode's window 0.
        transient_released: bool = false,
        /// (profile builds: `wide_profile.enabled`) The last wide call's JOINLESS merge (`MergeStats`); compiled out elsewhere.
        last_merge: if (wide_profile.enabled) MergeStats else void = if (wide_profile.enabled) .{} else {},
        /// The decode lane's wave stages, bound at construction (`Options.banked`): one group per bank (the stock route)
        /// or one banked group over every bank.
        gate_up_wave: *const GateUpWaveFn = gateUpWaveGrouped,
        down_wave: *const DownWaveFn = downWaveGrouped,
        /// Where a decode call commits the caller's hoist, bound at construction (`Options.hoist_first`): behind the
        /// hit wave (the stock route: `pre_barrier` commits nothing, `post_hit` commits the hoist), or right behind
        /// the routing barrier's arrays, before the host waits on them (HOIST_FIRST: `pre_barrier` commits both,
        /// `post_hit` nothing), so the GPU runs the hoist while the host wakes, routes and encodes the hit wave.
        pre_barrier: *const PreBarrierFn = preBarrierNone,
        post_hit: *const PostHitFn = postHitHoist,
        /// DEVROUTE (`Options.devroute`): per layer two resident LUTs (u32 [n_experts], host arrays written in place, kept),
        /// the one each layer's next decode call reads (`dev_parity`): a call reads one, its route's result is written
        /// into the other, so the host never writes a LUT a committed wave may still read. Empty without the route.
        devroute: bool = false,
        dev_lut: [][2]T = &.{},
        dev_parity: []u1 = &.{},
        /// A gated scheduler's decode call's miss parts, bound at construction (`bindMergeGateWaits`): each projection of
        /// each bank waited on its own (the stock route), or every array of one gate value in one wait (MERGE_GATE_WAITS).
        gated_parts: if (routes.gated) *const GatedPartsFn else void = if (routes.gated) gatedPartsStock else {},

        const GateUpWaveFn = fn (self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) anyerror!Wave;
        const GatedPartsFn = fn (self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void;
        const DownWaveFn = fn (self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) anyerror![]const T;
        const PreBarrierFn = fn (g: *G, barrier: []const T, hoist: []const T) anyerror!void;
        const PostHitFn = fn (g: *G, hoist: []const T) anyerror!void;

        /// `banked`: the math's banked route (its `has_banked`; the quant installed it) runs every wave as one group.
        /// `hoist_first`: HOIST_FIRST (exact: the same evals, only their commit order moves) commits a decode call's
        /// hoist right behind its routing barrier's arrays, before the barrier's wait, instead of behind the hit wave.
        /// `devroute`: DEVROUTE (exact): in decode, each call's hit wave is a device graph (the routed ids through the
        /// layer's resident LUT into the banked texts, every routed pair) committed behind the barrier's arrays, before the
        /// host's wait and route; the host then routes and builds only the miss parts; the join takes every hit pair from
        /// the device wave and every miss pair from its part. Needs the banked route.
        pub const Options = struct { gates: []const Gate = &.{}, event: ?expert_event.Event = null, wide: Wide = .{}, banked: bool = false, hoist_first: bool = false, devroute: bool = false };

        /// The wide lane's host scratch, reused across calls.
        const WideScratch = struct {
            ids: std.ArrayList(u16) = .empty,
            /// Per expert: its index in `distinct`, or -1.
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            /// Per expert: its rows in the call (the feed's order).
            count: std.ArrayList(u32) = .empty,
            cold_slot: std.ArrayList(u32) = .empty,
            cold_act: std.ArrayList(u32) = .empty,
            cold_pos: std.ArrayList(u32) = .empty,
            /// The unjoined route: a call's rows' routed positions, and its wave-ordered rows' call rows.
            call_pos: std.ArrayList(u32) = .empty,
            wave_pos: std.ArrayList(u32) = .empty,
            slot: std.ArrayList(u32) = .empty,
            act_row: std.ArrayList(u32) = .empty,
            pos: std.ArrayList(u32) = .empty,
            inv: std.ArrayList(u32) = .empty,
            kept: std.ArrayList(T) = .empty,
            /// JOINLESS: each assignment's (source, row), int32 pairs.
            loc: std.ArrayList(i32) = .empty,
            /// JOINLESS: the merge's plan (`mergeJoinless`).
            jl: JoinlessScratch(T) = .{},
            /// The deferred base-bank rows (`Wide.defer_base`): slots, act rows, routed positions.
            def_slot: std.ArrayList(u32) = .empty,
            def_act: std.ArrayList(u32) = .empty,
            def_pos: std.ArrayList(u32) = .empty,
            /// `Wide.route_cut`: the call's rows by group (`groupRows`; the rows themselves in `inv`, free until the join):
            /// per expert its group, the groups' row offsets.
            grp_of: std.ArrayList(u16) = .empty,
            grp_off: std.ArrayList(u32) = .empty,

            fn deinit(w: *WideScratch, a: std.mem.Allocator) void {
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.count, &w.cold_slot, &w.cold_act, &w.cold_pos, &w.call_pos, &w.wave_pos, &w.slot, &w.act_row, &w.pos, &w.inv, &w.kept, &w.loc, &w.def_slot, &w.def_act, &w.def_pos, &w.grp_of, &w.grp_off }) |l| l.deinit(a);
                w.jl.deinit(a);
            }
        };

        pub fn init(a: std.mem.Allocator, g: *G, source: *S, math: M, c: anytype) !Self {
            return initWith(a, g, source, math, c, .{});
        }

        pub fn initWith(a: std.mem.Allocator, g: *G, source: *S, math: M, c: anytype, opt: Options) !Self {
            // A decode-width forward must fit one route: its calls are the decode lane's by arithmetic.
            if (@as(u64, bound_decode_rows) * c.n_experts_per_tok > max_route_ids) return error.DecodeRowsWiderThanRoute;
            if (routes.lookahead and opt.gates.len != c.n_layers) return error.LookaheadNeedsGates;
            if (routes.gated and G == ops.MlxOps and opt.event == null) return error.GatedNeedsEvent;
            const wr = opt.wide;
            if (wr.depth < 1 or wr.depth > expert_stream.max_wide_depth) return error.InvalidWideRoute;
            if ((wr.seed or wr.hot_first or wr.depth > 1 or wr.cold_rows > 0) and !routes.prefill) return error.InvalidWideRoute;
            if (wr.cold_rows > Wide.max_cold_rows) return error.InvalidWideRoute;
            if (wr.seed and comptime !@hasDecl(S, "seedPrefill")) return error.InvalidWideRoute;
            if (wr.defer_base and (wr.cold_rows > 0 or !routes.prefill or comptime !@hasDecl(S, "holdBase"))) return error.InvalidWideRoute;
            if (wr.read_ahead and (!wr.seed or comptime !@hasDecl(S, "readAheadSeed"))) return error.InvalidWideRoute;
            if (wr.base_at_seed and (!wr.seed or !wr.defer_base)) return error.InvalidWideRoute;
            if (wr.seed_aligned and (!wr.base_at_seed or !wr.hot_first or comptime !@hasDecl(S, "seedRanks"))) return error.InvalidWideRoute;
            if (opt.banked and comptime !hasBanked()) return error.BankedNotInMath;
            if (wr.depth > 1) {
                if (comptime @hasDecl(S, "wideDepth")) {
                    if (source.wideDepth() < wr.depth) return error.WideDepthExceedsSource;
                } else return error.WideDepthExceedsSource;
            }
            const banks = try a.alloc([n_banks]?Arrays, c.n_layers);
            errdefer a.free(banks);
            for (banks, 0..) |*b, l| {
                b.* = @splat(null);
                for ([_]BankKind{ .base, .transient }) |kind| b[@backingInt(kind)] = try bind(g, source, @intCast(l), kind);
            }
            var self: Self = .{ .a = a, .source = source, .math = math, .hidden = @intCast(c.hidden_size), .n_experts = c.n_routed_experts, .banks = banks, .gates = opt.gates, .g = g, .wide_route = wr };
            if (opt.event) |e| self.event = e;
            if (comptime hasBanked()) {
                if (opt.banked) {
                    self.gate_up_wave = gateUpWaveBanked;
                    self.down_wave = downWaveBanked;
                }
            }
            if (opt.hoist_first) {
                self.pre_barrier = preBarrierHoist;
                self.post_hit = postHitNone;
            }
            if (opt.devroute) {
                if (!opt.banked) return error.DevrouteNeedsBanked;
                if (comptime !(@hasDecl(S, "residentLut") and @hasDecl(S, "decoding"))) return error.DevrouteNotInSource;
                self.devroute = true;
            }
            return self;
        }

        /// DEVROUTE: both LUTs of every layer written from the grown policies (the decode's first calls read them).
        fn devLutsAtGrow(self: *Self, g: *G) !void {
            if (comptime !(@hasDecl(S, "residentLut") and @hasDecl(S, "decoding"))) {
                return error.DevrouteNotInSource;
            } else return self.devLutsAtGrowIn(g);
        }

        fn devLutsAtGrowIn(self: *Self, g: *G) !void {
            if (self.dev_lut.len == 0) {
                const n_l = self.banks.len;
                self.dev_lut = try self.a.alloc([2]T, n_l);
                self.dev_parity = try self.a.alloc(u1, n_l);
                const zeros = try self.a.alloc(u32, self.n_experts);
                defer self.a.free(zeros);
                @memset(zeros, 0);
                for (self.dev_lut) |*pair| for (pair) |*x| {
                    x.* = g.keep(try g.hostArray(std.mem.sliceAsBytes(zeros), &.{@intCast(self.n_experts)}, .uint32));
                };
            }
            for (self.dev_lut, self.dev_parity, 0..) |pair, *par, l| {
                for (pair) |x| self.source.residentLut(@intCast(l), std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(x)))));
                par.* = 0;
            }
        }

        /// DEVROUTE: layer `layer`'s LUT for its next call, from the policy after this call's route; that call reads it.
        fn devLutAfterRoute(self: *Self, g: *G, layer: u32) !void {
            if (comptime !(@hasDecl(S, "residentLut") and @hasDecl(S, "decoding"))) {
                return error.DevrouteNotInSource;
            } else return self.devLutAfterRouteIn(g, layer);
        }

        fn devLutAfterRouteIn(self: *Self, g: *G, layer: u32) !void {
            const next = self.dev_parity[layer] +% 1;
            self.source.residentLut(layer, std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(self.dev_lut[layer][next])))));
            self.dev_parity[layer] = next;
        }

        /// DEVROUTE: the call's every routed pair through the banked texts, the slots from the layer's LUT on the device
        /// (hits: their resident (bank, row), the host's plan's same; misses: base row 0, never joined); `[n * k, hidden]`
        /// in routed order.
        fn devWave(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            if (comptime !hasBanked()) {
                return error.DevrouteNeedsBanked;
            } else return self.devWaveIn(g, layer, xf, indices, n, k);
        }

        fn devWaveIn(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            const nk: c_int = @intCast(n * k);
            const lut = self.dev_lut[layer][self.dev_parity[layer]];
            const ids = try g.take(lut, try g.reshape(indices, &.{nk}), 0);
            var tok: [max_route_ids]i32 = undefined;
            for (tok[0 .. n * k], 0..) |*t, p| t.* = @intCast(p / k);
            const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0 .. n * k]), &.{nk}, .int32), 0);
            const banks = try self.bankedArrays(layer, null);
            return self.math.downBanked(g, try self.math.gateUpBanked(g, x, ids, &banks), ids, &banks);
        }

        /// The stock route: the barrier's wait commits its own arrays; nothing is committed ahead of it.
        fn preBarrierNone(_: *G, _: []const T, _: []const T) !void {}

        /// HOIST_FIRST: the barrier's arrays in one commit (the event the wait reads signals at its end), then the hoist
        /// in its own commit behind it, so the wait returns at the router's end while the GPU runs the hoist. A call
        /// with no hoist commits the barrier's arrays alone (the wait that follows is then only a wait).
        fn preBarrierHoist(g: *G, barrier: []const T, hoist: []const T) !void {
            try g.asyncEval(barrier);
            if (hoist.len > 0) try g.asyncEval(hoist);
        }

        /// The stock route: the hoist is committed behind the hit wave, ahead of every miss wave.
        fn postHitHoist(g: *G, hoist: []const T) !void {
            if (hoist.len > 0) try g.asyncEval(hoist);
        }

        /// HOIST_FIRST: the hoist was committed before the barrier's wait.
        fn postHitNone(_: *G, _: []const T) !void {}

        /// HOIST_FIRST as bound at construction.
        pub fn hoistFirst(self: *const Self) bool {
            return self.pre_barrier == &preBarrierHoist;
        }

        fn hasBanked() bool {
            return @hasDecl(M, "has_banked") and M.has_banked;
        }

        /// The router that scores layer `layer`'s read-ahead: the NEXT layer's
        /// (none after the last layer).
        pub fn predictorGate(self: *const Self, layer: u32) ?Gate {
            return if (layer + 1 < self.gates.len) self.gates[layer + 1] else null;
        }

        /// X19: a math that binds its predictor at construction (MiMo's `predict_router`) scores through its own
        /// instance; every other math keeps `nextScores`.
        fn scoresOf(self: *const Self, g: *G, xf: T, gate: Gate) !T {
            if (comptime @hasDecl(M, "predictorScoresBound")) return self.math.predictorScoresBound(g, xf, gate.w, gate.bias);
            return nextScores(g, xf, gate);
        }

        /// The architecture's predictor, or DeepSeek's sqrt(softplus(f32(x @ w^T))) + bias.
        fn nextScores(g: *G, xf: T, gate: Gate) !T {
            if (comptime @hasDecl(M, "predictorScores")) return M.predictorScores(g, xf, gate.w, gate.bias);
            const z = try g.astype(try g.matmul(xf, try g.transpose(gate.w)), .float32);
            return g.add(try g.sqrt(try g.logaddexp(z, try g.scalar(0, .float32))), gate.bias);
        }

        /// An event wait's aliases of `xs` (the GPU reads them after `value`), into `outs` (any contents). One wait (one
        /// pass-through primitive, one encoder split) whatever `xs.len`; the trace backend counts the waits (`wait_calls`).
        fn eventWait(self: *Self, g: *G, xs: []const T, value: u64, deps: []const T, outs: []T) !void {
            if (G == ops.MlxOps) {
                try mlxEventWait(g, self.event, xs, value, deps, outs);
            } else {
                if (comptime @hasField(G, "wait_calls")) g.wait_calls += 1;
                for (xs, outs) |x, *o| o.* = try g.eventAlias(x, value, deps.len);
            }
        }

        fn waitProj(self: *Self, g: *G, p: Projection(T), value: u64, deps: []const T) !Projection(T) {
            const xs = EventProjection.inputs(p);
            var outs: [EventProjection.count]T = undefined;
            try self.eventWait(g, &xs, value, deps, &outs);
            return EventProjection.fromWaited(outs);
        }

        pub fn deinit(self: *Self) void {
            for (self.dev_lut) |pair| for (pair) |x| self.g.release(x);
            self.a.free(self.dev_lut);
            self.a.free(self.dev_parity);
            self.wide.deinit(self.a);
            self.a.free(self.banks);
            self.* = undefined;
        }

        /// A bank with rows must have arrays the math can bind (MLX slot memory).
        fn bind(g: *G, source: *S, layer: u32, kind: BankKind) !?Arrays {
            const arrays = try source.bankArrays(g, layer, kind);
            if (arrays == null and source.bankRows(layer, kind) > 0) return error.SlotArraysUnbound;
            return arrays;
        }

        /// The phase change's first free: every layer's transient binding nulled (from here to `grow` a stray use
        /// fails on null, never on freed arrays), then the source frees the scratch. Returns the bytes freed. A
        /// refusal leaves the bindings null (fails closed; the Module refuses every later request).
        pub fn releaseTransient(self: *Self) !u64 {
            for (self.banks) |*b| b[@backingInt(BankKind.transient)] = null;
            self.transient_released = true;
            return self.source.releaseTransient();
        }

        /// The one phase change: the source grows, the grown rows (and after a release, decode's window 0) are bound.
        pub fn grow(self: *Self, g: *G, decode_rows: []const u32) !void {
            try self.source.grow(decode_rows);
            for (self.banks, 0..) |*b, l| {
                b[@backingInt(BankKind.ext)] = try bind(g, self.source, @intCast(l), .ext);
                if (self.transient_released) b[@backingInt(BankKind.transient)] = try bind(g, self.source, @intCast(l), .transient);
            }
            self.transient_released = false;
            if (self.devroute) try self.devLutsAtGrow(g);
        }

        /// DEVROUTE as bound at construction.
        /// Construction: the parts' combine reads up to `bound` sources in place (`runWideParts` merges only above it).
        pub fn setJoinlessBound(self: *Self, bound: usize) void {
            self.joinless_bound = @max(bound, joinless_sources);
        }

        pub fn devRoute(self: *const Self) bool {
            return self.devroute;
        }

        /// The reverse phase change's free: the grown rows' and window 0's bindings nulled (a stray use fails on null),
        /// then the source frees them, the prompt rows' residents forgotten or kept (`residents`; the base bindings stay).
        /// Returns the bytes freed.
        pub fn shrink(self: *Self, prompt_rows: []const u32, residents: expert_stream.Residents) !u64 {
            for (self.banks) |*b| {
                b[@backingInt(BankKind.ext)] = null;
                b[@backingInt(BankKind.transient)] = null;
            }
            self.transient_released = true;
            return self.source.shrink(prompt_rows, residents);
        }

        /// The reverse phase change's allocation: the prompt's scratch re-created and bound. Returns its bytes (0 when
        /// the scratch stayed through decode: the release route off).
        pub fn regrowTransient(self: *Self, g: *G, released: bool) !u64 {
            const bytes: u64 = if (released) try self.source.regrowTransient() else 0;
            for (self.banks, 0..) |*b, l| b[@backingInt(BankKind.transient)] = try bind(g, self.source, @intCast(l), .transient);
            self.transient_released = false;
            return bytes;
        }

        /// After the forward's last eval: settles and unpins released calls.
        pub fn flush(self: *Self) !void {
            try self.source.flush();
        }

        pub const Hook = struct {
            ex: *Self,
            layer: u32,

            pub fn routed(h: Hook, g: *G, xf: T, indices: T) !T {
                return h.ex.run(g, h.layer, xf, indices, &.{});
            }

            /// `routed`, with the caller's arrays that do not wait on this call (`hoist`: the layer's
            /// shared expert and HC tail) committed once its reads are issued and its hit wave
            /// started: they run during the read wait, not in the next routing barrier.
            pub fn routedHoist(h: Hook, g: *G, xf: T, indices: T, hoist: []const T) !T {
                return h.ex.run(g, h.layer, xf, indices, hoist);
            }

            /// `routedHoist` ending before the join's take (JOIN_INDEXED, a decode-lane call): the concatenated outputs and
            /// each routed position's row in them, for a combine that reads the rows in place.
            pub fn routedHoistIndexed(h: Hook, g: *G, xf: T, indices: T, hoist: []const T) !Indexed {
                return h.ex.runEnd(g, h.layer, xf, indices, hoist, .indexed);
            }

            /// JOINLESS (a wide call only: more than max_route_ids ids): the unjoined outputs as the
            /// combine's sources and each assignment's (source, row); `releaseParts` after the combines are evaluated.
            pub fn routedParts(h: Hook, g: *G, xf: T, indices: T) !Parts {
                if (comptime !routes.prefill) return error.PrefillLaneNotPorted;
                const n: u32 = @intCast(g.shapeOf(xf).dim(0));
                const k: u32 = @intCast(g.shapeOf(indices).dim(1));
                if (comptime !@hasDecl(M, "prefill_min_tokens")) {
                    if (n * k <= max_route_ids) return error.WideLaneUnderMinIds;
                }
                return h.ex.runWideParts(g, h.layer, xf, indices, n, k);
            }

            pub fn releaseParts(h: Hook, g: *G) void {
                h.ex.releaseParts(g);
            }

            /// (profile builds) The last wide call's JOINLESS merge; zeros elsewhere.
            pub fn lastMerge(h: Hook) MergeStats {
                return if (comptime wide_profile.enabled) h.ex.last_merge else .{};
            }

            /// P1 (`Wide.read_ahead`, fixed at construction): the model runs its predictor pass for this layer.
            pub fn readAhead(h: Hook) bool {
                return h.ex.wide_route.read_ahead;
            }

            /// The source is in its prompt phase (before its one phase change): the forward is a prompt's, so its predictor
            /// pass has rows to read ahead into. A source without a phase API is always prompting.
            pub fn prompting(h: Hook) bool {
                return if (comptime @hasDecl(S, "decoding")) !h.ex.source.decoding() else true;
            }

            /// P1: the layer's predicted seed, experts hottest first, read ahead of its routed call.
            pub fn readAheadSeed(h: Hook, experts: []const u16) !void {
                if (comptime !@hasDecl(S, "readAheadSeed")) return error.ReadAheadNotStreamed;
                return h.ex.source.readAheadSeed(h.layer, experts);
            }
        };

        pub fn at(self: *Self, layer: u32) Hook {
            return .{ .ex = self, .layer = layer };
        }

        const Group = struct { bank: BankKind, n: u32 = 0, pos: [max_route_ids]u8 = undefined };

        /// Outputs in wave order and the routed position of each output row.
        /// `pos` holds each concatenated output row's routed position; DEVROUTE's device wave adds a row for every pair
        /// before the miss parts' rows (2 x the route's ids).
        const Acc = struct {
            outs: [max_route_ids]T = undefined,
            n_outs: usize = 0,
            pos: [2 * max_route_ids]u32 = undefined,
            n_pos: usize = 0,
        };

        const Wave = struct {
            groups: [n_banks]Group = undefined,
            n: usize = 0,
            ids: [n_banks]T = undefined,
            h: [n_banks]T = undefined,
        };

        /// The positions of wave `w`, grouped by bank in first-appearance order
        /// (`Exl3PackedOps.gate_up`'s groups), each group's gate/up + SwiGLU.
        fn gateUpWaveGrouped(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) anyerror!Wave {
            var wave: Wave = .{};
            for (sv.waves, sv.refs, 0..) |wv, ref, pos| {
                if (wv != w) continue;
                const gi = for (wave.groups[0..wave.n], 0..) |gr, i| {
                    if (gr.bank == ref.bank) break i;
                } else blk: {
                    wave.groups[wave.n] = .{ .bank = ref.bank };
                    wave.n += 1;
                    break :blk wave.n - 1;
                };
                const gr = &wave.groups[gi];
                gr.pos[gr.n] = @intCast(pos);
                gr.n += 1;
            }
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]).?;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |pos, *t, *r| {
                    t.* = @intCast(pos / k);
                    r.* = sv.refs[pos].row;
                }
                const n: c_int = @intCast(gr.n);
                const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{n}, .int32), 0);
                wave.ids[i] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{n}, .uint32);
                wave.h[i] = try self.math.gateUp(g, x, wave.ids[i], arrays.gate, arrays.up);
            }
            return wave;
        }

        /// Each group's down; the outputs join the call's accumulator.
        fn downWaveGrouped(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) anyerror![]const T {
            const first = acc.n_outs;
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]).?;
                acc.outs[acc.n_outs] = try self.math.down(g, wave.h[i], wave.ids[i], arrays.down);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |pos| {
                    acc.pos[acc.n_pos] = pos;
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

        /// A wave's bank arrays for the banked texts: each bank's; a bank without arrays (no rows bound, or not waited
        /// in a gated part) is given another bank's (the packed ids never name it).
        fn bankedArrays(self: *Self, layer: u32, over: ?*const [n_banks]?Arrays) ![n_banks]Arrays {
            const src = if (over) |o| o else &self.banks[layer];
            const fill = for (src) |b| {
                if (b) |arr| break arr;
            } else return error.BankedWaveWithoutArrays;
            var out: [n_banks]Arrays = undefined;
            for (src, &out) |b, *o| o.* = b orelse fill;
            return out;
        }

        /// The banked route's gate/up: wave `w`'s positions as one group in routed order, ids packed (bank << 24 | slot row).
        fn gateUpWaveBanked(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) anyerror!Wave {
            var wave: Wave = .{};
            const gr = &wave.groups[0];
            gr.* = .{ .bank = .base };
            var tok: [max_route_ids]i32 = undefined;
            var rows: [max_route_ids]u32 = undefined;
            for (sv.waves, sv.refs, 0..) |wv, ref, pos| {
                if (wv != w) continue;
                gr.pos[gr.n] = @intCast(pos);
                tok[gr.n] = @intCast(pos / k);
                rows[gr.n] = (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
                gr.n += 1;
            }
            if (gr.n == 0) return wave;
            wave.n = 1;
            const banks = try self.bankedArrays(layer, over);
            const n: c_int = @intCast(gr.n);
            const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{n}, .int32), 0);
            wave.ids[0] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{n}, .uint32);
            wave.h[0] = try self.math.gateUpBanked(g, x, wave.ids[0], &banks);
            return wave;
        }

        /// The banked route's down: the wave's one group over every bank.
        fn downWaveBanked(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) anyerror![]const T {
            const first = acc.n_outs;
            if (wave.n == 0) return acc.outs[first..first];
            const banks = try self.bankedArrays(layer, over);
            acc.outs[acc.n_outs] = try self.math.downBanked(g, wave.h[0], wave.ids[0], &banks);
            acc.n_outs += 1;
            for (wave.groups[0].pos[0..wave.groups[0].n]) |pos| {
                acc.pos[acc.n_pos] = pos;
                acc.n_pos += 1;
            }
            return acc.outs[first..acc.n_outs];
        }

        /// The gated waves on the route bound at construction (`gated_parts`).
        fn gatedParts(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) !void {
            return self.gated_parts(self, g, layer, xf, k, sv, gates, acc);
        }

        /// MERGE_GATE_WAITS bound (a decode call's gated miss parts wait once per gate value), once at construction,
        /// before any call. Needs the gated route.
        pub fn bindMergeGateWaits(self: *Self) !void {
            if (comptime routes.gated) {
                self.gated_parts = gatedPartsMerged;
            } else return error.MergeGateWaitsNeedsGated;
        }

        /// MERGE_GATE_WAITS as bound at construction.
        pub fn mergeGateWaits(self: *const Self) bool {
            return if (comptime routes.gated) self.gated_parts == &gatedPartsMerged else false;
        }

        /// The gated waves (the lookahead4 lane's event gate): every part's
        /// gate/up over bank arrays waited at `gu`, then part p's down over
        /// arrays waited at `down_first + p`, ordered after every gate/up and
        /// the previous part's down. Nothing waits on the host. Each projection
        /// of each bank waits on its own (the stock route).
        fn gatedPartsStock(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void {
            var gu: [n_banks]?Arrays = @splat(null);
            for (sv.waves, sv.refs) |w, ref| {
                const b = @backingInt(ref.bank);
                if (w == 0 or gu[b] != null) continue;
                const arrays = self.banks[layer][b].?;
                gu[b] = .{ .gate = try self.waitProj(g, arrays.gate, gates.gu, &.{}), .up = try self.waitProj(g, arrays.up, gates.gu, &.{}), .down = arrays.down };
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), &gu);
                for (waves[p].h[0..waves[p].n]) |h| {
                    hs[n_hs] = h;
                    n_hs += 1;
                }
            }
            var prev: []const T = &.{};
            for (0..sv.n_parts) |p| {
                var deps: [2 * max_route_ids]T = undefined;
                @memcpy(deps[0..n_hs], hs[0..n_hs]);
                @memcpy(deps[n_hs..][0..prev.len], prev);
                var dn = gu;
                for (&dn) |*d| if (d.*) |*arr| {
                    arr.down = try self.waitProj(g, arr.down, gates.down_first + p, deps[0 .. n_hs + prev.len]);
                };
                prev = try self.down_wave(self, g, layer, &waves[p], acc, &dn);
            }
        }

        /// MERGE_GATE_WAITS: `gatedPartsStock`'s waves over the same arrays, gates and deps, with one event wait per gate
        /// value: the gate and up arrays of every bank the call's misses read wait at `gu` together, then part p's down
        /// arrays of every such bank at `down_first + p` together (behind the same deps). The stock route waits each
        /// projection of each bank on its own at those same values. Exact: every array aliases the same buffer behind the
        /// same value; only the waits (each one pass-through primitive and one encoder split) are fewer.
        fn gatedPartsMerged(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void {
            const pc = EventProjection.count;
            var gu: [n_banks]?Arrays = @splat(null);
            // The banks the misses read, in the stock route's order (first appearance), each one's gate then up arrays.
            var order: [n_banks]usize = undefined;
            var nb: usize = 0;
            var seen: [n_banks]bool = @splat(false);
            var xs: [2 * n_banks * pc]T = undefined;
            for (sv.waves, sv.refs) |w, ref| {
                const b = @backingInt(ref.bank);
                if (w == 0 or seen[b]) continue;
                seen[b] = true;
                const arrays = self.banks[layer][b].?;
                xs[2 * pc * nb ..][0..pc].* = EventProjection.inputs(arrays.gate);
                xs[2 * pc * nb + pc ..][0..pc].* = EventProjection.inputs(arrays.up);
                order[nb] = b;
                nb += 1;
            }
            if (nb > 0) {
                var outs: [2 * n_banks * pc]T = undefined;
                try self.eventWait(g, xs[0 .. 2 * pc * nb], gates.gu, &.{}, outs[0 .. 2 * pc * nb]);
                for (order[0..nb], 0..) |b, i| {
                    const arrays = self.banks[layer][b].?;
                    gu[b] = .{ .gate = EventProjection.fromWaited(outs[2 * pc * i ..][0..pc].*), .up = EventProjection.fromWaited(outs[2 * pc * i + pc ..][0..pc].*), .down = arrays.down };
                }
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), &gu);
                for (waves[p].h[0..waves[p].n]) |h| {
                    hs[n_hs] = h;
                    n_hs += 1;
                }
            }
            var prev: []const T = &.{};
            for (0..sv.n_parts) |p| {
                var deps: [2 * max_route_ids]T = undefined;
                @memcpy(deps[0..n_hs], hs[0..n_hs]);
                @memcpy(deps[n_hs..][0..prev.len], prev);
                var dn = gu;
                var dx: [n_banks * pc]T = undefined;
                var nd: usize = 0;
                for (dn) |d| if (d) |arr| {
                    dx[pc * nd ..][0..pc].* = EventProjection.inputs(arr.down);
                    nd += 1;
                };
                if (nd > 0) {
                    var dout: [n_banks * pc]T = undefined;
                    try self.eventWait(g, dx[0 .. pc * nd], gates.down_first + p, deps[0 .. n_hs + prev.len], dout[0 .. pc * nd]);
                    var i: usize = 0;
                    for (&dn) |*d| if (d.*) |*arr| {
                        arr.down = EventProjection.fromWaited(dout[pc * i ..][0..pc].*);
                        i += 1;
                    };
                }
                prev = try self.down_wave(self, g, layer, &waves[p], acc, &dn);
            }
        }

        /// `PackedDecode.run` (gate/up publish before down): the routing
        /// barrier, the residents' wave, then per miss part gate/up after
        /// `waitGu` and down after `waitDown`, each wave started on the GPU;
        /// release; the outputs in the router's order, `[n, k, hidden]` f32.
        /// A residency query over the source (before the call's route admits and evicts; profile builds).
        const ResidentAt = struct {
            src: *S,
            layer: u32,
            fn of(r: ResidentAt, e: u16) bool {
                return r.src.isResident(r.layer, e);
            }
        };

        /// The stream's read-busy gauge (ns), for the profile builds' timeline.
        fn readGauge(self: *const Self) u64 {
            return if (comptime @hasDecl(S, "readWallNs")) self.source.readWallNs() else 0;
        }

        pub fn run(self: *Self, g: *G, layer: u32, xf: T, indices: T, hoist: []const T) !T {
            return self.runEnd(g, layer, xf, indices, hoist, .take);
        }

        /// A decode-lane call's outputs before the join's take (`Hook.routedHoistIndexed`, JOIN_INDEXED: the caller's
        /// combine reads the rows in place): the concatenated outputs [rows, hidden] and each routed position's row in
        /// them, u32 [n * k] (`joinOrder`; the stock join takes exactly these rows).
        pub const Indexed = struct { rows: T, order: T };

        /// How a call ends: the join's take (`run`, every caller's route) or the indexed rows (JOIN_INDEXED).
        const Ending = enum { take, indexed };

        fn runEnd(self: *Self, g: *G, layer: u32, xf: T, indices: T, hoist: []const T, comptime ending: Ending) !(if (ending == .take) T else Indexed) {
            const n: u32 = @intCast(g.shapeOf(xf).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            const n_ids = n * k;
            // A math with its own prefill token threshold routes by logical M.
            // DeepSeek keeps its decode / verify route through 48 top-6 ids.
            if (if (comptime @hasDecl(M, "prefill_min_tokens")) n >= M.prefill_min_tokens else n_ids > max_route_ids) {
                // Only a decode-width layer hoists (the trunk's small-rows segments).
                std.debug.assert(hoist.len == 0);
                if (comptime ending == .indexed) {
                    // The indexed join serves the decode lane only (a wide call joins in its combine already).
                    return error.IndexedJoinWideCall;
                } else if (comptime !routes.prefill) {
                    return error.PrefillLaneNotPorted;
                } else {
                    return self.runWide(g, layer, xf, indices, n, k);
                }
            }
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [expert_lookahead.max_rows * 512]f32 = undefined;
            var scores: []const f32 = &.{};
            var tt = dt.now();
            // A0 (profile builds): the call's barrier, route and MoE build (`first_cycle.call`).
            const t_call = tt;
            if (comptime timeline.enabled) timeline.point(layer, .call, self.readGauge(), false);
            // The installed decode-only predictor follows the source's phase; the last layer predicts nothing.
            const predict = if (comptime routes.lookahead and predictor_decode_only) self.source.decoding() else routes.lookahead;
            const sc: ?T = if (if (predict) self.predictorGate(layer) else null) |gate| try self.scoresOf(g, xf, gate) else null;
            // The barrier's arrays: the ids, then the scores when there are any.
            var bar: [2]T = undefined;
            var n_bar: usize = 0;
            for ([_]?T{ indices, sc }) |x| if (x) |v| {
                bar[n_bar] = v;
                n_bar += 1;
            };
            // DEVROUTE (decode): the barrier's arrays committed, then the device hit wave behind them (before the host waits).
            const dev: ?T = if (comptime @hasDecl(S, "decoding")) (if (self.devroute and self.source.decoding()) try self.devWave(g, layer, xf, indices, n, k) else null) else null;
            if (dev) |d| {
                try g.asyncEval(bar[0..n_bar]);
                try g.asyncEval(&.{d});
            }
            // HOIST_FIRST commits them and the hoist here (the evals below then only wait); the stock route nothing.
            try self.pre_barrier(g, bar[0..n_bar], hoist);
            if (n_bar > 1) try g.evalAll(bar[0..n_bar]);
            if (sc) |s| scores = try g.hostF32(s, score_buf[0 .. n * self.n_experts]);
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            tt = dt.charge(.barrier, tt);
            const t_barrier = tt;
            if (comptime timeline.enabled) timeline.point(layer, .barrier, 0, false);
            // A0 (profile builds): the read gauge at the barrier's end; in the first cycle the tail set's reads (before
            // the route plans); in the prompt the tail set itself (a decode-width prompt call).
            const wall_b: u64 = if (comptime first_cycle.enabled and @hasDecl(S, "readWallNs")) self.source.readWallNs() else 0;
            const tail_reads: u64 = if (comptime first_cycle.enabled and @hasDecl(S, "isResident")) first_cycle.tailReads(layer, ResidentAt{ .src = self.source, .layer = layer }, ResidentAt.of) else 0;
            if (comptime first_cycle.enabled) {
                first_cycle.tailRecord(layer, ids, n, k);
                tt = dt.now();
            }
            const t_route0 = tt;
            const call = try self.source.route(layer, ids, scores);
            const t_route = dt.charge(.route, tt);
            dt.countCall();
            if (comptime timeline.enabled) timeline.point(layer, .route, 0, false);
            var released = false;
            errdefer if (!released) self.source.release(call);
            const sv = self.source.served(call);
            var acc: Acc = .{};
            if (dev) |d| {
                // DEVROUTE: every pair's row from the device wave; the miss parts' rows follow and replace theirs in the join.
                acc.outs[0] = d;
                acc.n_outs = 1;
                for (0..n_ids) |p| acc.pos[p] = @intCast(p);
                acc.n_pos = n_ids;
            } else {
                // Residents: gate/up and down at once.
                const hits = try self.gate_up_wave(self, g, layer, xf, k, sv, 0, null);
                if (hits.n > 0) try g.asyncEval(try self.down_wave(self, g, layer, &hits, &acc, null));
            }
            // VERIFY_ENCODE hoist: behind the hit wave, ahead of every miss wave (HOIST_FIRST: committed before the barrier's wait).
            try self.post_hit(g, hoist);
            if (comptime timeline.enabled) timeline.point(layer, .hit, 0, false);
            if (routes.gated) {
                if (try self.source.gate(call)) |gates| {
                    if (comptime timeline.enabled) timeline.gate(layer, gates.gu, gates.down_first, gates.n_parts);
                    try self.gatedParts(g, layer, xf, k, sv, gates, &acc);
                }
            } else for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                tt = dt.now();
                try self.source.waitGu(call, part);
                _ = dt.charge(.read_wait, tt);
                const wave = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), null);
                try g.asyncEval(wave.h[0..wave.n]);
                tt = dt.now();
                try self.source.waitDown(call, part);
                _ = dt.charge(.read_wait, tt);
                try g.asyncEval(try self.down_wave(self, g, layer, &wave, &acc, null));
            }
            self.source.release(call);
            released = true;
            // DEVROUTE: the layer's next call reads the policy after this route (the other LUT of the pair).
            if (dev != null) try self.devLutAfterRoute(g, layer);
            if (comptime timeline.enabled) timeline.point(layer, .end, self.readGauge(), sv.n_parts > 0);
            if (comptime first_cycle.enabled) first_cycle.call(layer, t_barrier - t_call, t_route - t_route0, dt.now() - t_route, ids, sv.waves, wall_b, tail_reads);
            return if (comptime ending == .take) self.join(g, &acc, n, k) else joinIndexed(g, &acc, n, k);
        }

        /// Each routed position's row in the concatenated outputs: the LAST row carrying it (DEVROUTE's miss parts follow
        /// the device wave and replace its rows for their pairs; the stock route carries each position once).
        fn joinOrder(pos: []const u32, n_ids: u32, out: *[max_route_ids]u32) u32 {
            for (pos, 0..) |p, j| out[p] = @intCast(j);
            return n_ids;
        }

        /// `take(concatenate(outputs), argsort(positions))`: the inverse permutation of the (unique)
        /// positions, made on the host; `[n, k, hidden]`.
        fn join(self: *const Self, g: *G, acc: *const Acc, n: u32, k: u32) !T {
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            const n_ids = joinOrder(acc.pos[0..acc.n_pos], n * k, &order);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// `join` without its take: the concatenated outputs and the order the take would read them in (JOIN_INDEXED).
        fn joinIndexed(g: *G, acc: *const Acc, n: u32, k: u32) !Indexed {
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            const n_ids = joinOrder(acc.pos[0..acc.n_pos], n * k, &order);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return .{ .rows = joined, .order = ord };
        }

        /// The self-check's joined row as the host reads it: a math with its own projection (bf16 joins) casts to
        /// f32; DeepSeek's EXL3 joins are f32 already and are read as they are.
        fn checkRow(g: *G, x: T) !T {
            return if (comptime @hasDecl(M, "Projection")) g.astype(x, .float32) else x;
        }

        /// `checkGated` on layer 0's first `top_k` experts and one fixed bf16 row.
        pub fn checkGates(self: *Self, g: *G, a: std.mem.Allocator, top_k: u32) !void {
            return self.checkGatesAt(g, a, top_k, 0);
        }

        /// `checkGated` on the first routed layer's first `top_k` experts (cold at construction: every one a miss)
        /// and one fixed bf16 row.
        pub fn checkGatesAt(self: *Self, g: *G, a: std.mem.Allocator, top_k: u32, first_layer: u32) !void {
            const hidden: usize = @intCast(self.hidden);
            var ids: [max_route_ids]u16 = undefined;
            for (ids[0..top_k], 0..) |*d, i| d.* = @intCast(i);
            const row = try a.alloc(u16, hidden);
            defer a.free(row);
            for (row, 0..) |*v, i| v.* = ops.bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 97)) - 48)) / 64.0);
            const kh = @as(usize, top_k) * hidden;
            const outs = try a.alloc(f32, 2 * kh);
            defer a.free(outs);
            const m = g.mark();
            defer g.resetTo(m);
            const xf = try g.hostArray(std.mem.sliceAsBytes(row), &.{ 1, self.hidden }, .bfloat16);
            try self.checkGated(g, first_layer, xf, ids[0..top_k], outs[0..kh], outs[kh..]);
        }

        /// P1's construction self-check (`Wide.read_ahead`), once on the real stream before any request:
        /// layer 0's records read ahead equal their demand reads, bit for bit, over two pool jobs.
        /// Returns how many records were checked.
        pub fn checkReadAhead(self: *Self) !u32 {
            if (comptime !@hasDecl(S, "checkReadAhead")) return error.ReadAheadNotStreamed;
            return self.source.checkReadAhead(0, expert_io.max_items + 1);
        }

        /// LOOKAHEAD4's construction check, once on the real stream before any request: one call of
        /// `ids` (one row `xf`) over `layer`, whose misses' waves wait on event gates and are
        /// evaluated (the GPU held until the pool publishes their bytes), then the same slots' waves
        /// with no gate after the bytes landed (the host-waits arm's). The two outputs must match bit
        /// for bit (`gated` / `waited`: `k * hidden` f32 each) and the pool must have forced no gate.
        pub fn checkGated(self: *Self, g: *G, layer: u32, xf: T, ids: []const u16, gated: []f32, waited: []f32) !void {
            if (comptime !routes.gated) @compileError("checkGated needs the gated route");
            const k: u32 = @intCast(ids.len);
            const m = g.mark();
            defer g.resetTo(m);
            const call = try self.source.route(layer, ids, &.{});
            var released = false;
            errdefer if (!released) self.source.release(call);
            const sv = self.source.served(call);
            const gates = (try self.source.gate(call)) orelse return error.GateCheckNoMiss;
            var acc: Acc = .{};
            const hits = try self.gate_up_wave(self, g, layer, xf, k, sv, 0, null);
            if (hits.n > 0) _ = try self.down_wave(self, g, layer, &hits, &acc, null);
            try self.gatedParts(g, layer, xf, k, sv, gates, &acc);
            _ = try g.hostF32(try checkRow(g, try self.join(g, &acc, 1, k)), gated);
            var ref: Acc = .{};
            if (hits.n > 0) _ = try self.down_wave(self, g, layer, &hits, &ref, null);
            for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                try self.source.waitGu(call, part);
                const wave = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), null);
                try self.source.waitDown(call, part);
                _ = try self.down_wave(self, g, layer, &wave, &ref, null);
            }
            _ = try g.hostF32(try checkRow(g, try self.join(g, &ref, 1, k)), waited);
            self.source.release(call);
            released = true;
            try self.source.flush();
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(gated), std.mem.sliceAsBytes(waited))) return error.GateCheckMismatch;
        }

        /// The wide lane: the routing barrier; the call's distinct experts (first
        /// appearance) routed in groups of at most `max_route_ids`; per group,
        /// every part waited, then per bank the group's rows through the layer's
        /// DIG-X route (slot = the row the source serves the expert in, act row =
        /// its token), drained and evaluated before the group is released (the
        /// next route may refill those slots); the outputs joined in the
        /// router's order, `[n, k, hidden]` f32. The DIG kernels read bf16
        /// activations (the lane of record's MoE input): another dtype is
        /// rounded to bf16 once, here. `Options.wide.seed` seeds the layer from
        /// the call and drains each group once; `.hot_first` orders the groups hottest first;
        /// `Options.wide.depth` 2 routes group g + 1 (its reads) before group g's waves.
        fn runWide(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            try self.runWideCore(g, layer, xf, indices, n, k, false);
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            // `take(concatenate(outputs), argsort(positions))`, the permutation made on the host.
            const joined = try g.concat(w.kept.items, 0);
            for (w.kept.items) |x| g.release(x);
            w.kept.clearRetainingCapacity();
            try w.inv.resize(a, n_ids);
            invertPositions(w.pos.items, w.inv.items);
            const ord = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// JOINLESS: the wide call's outputs unjoined (above `max_parts`, the smallest merged into one source:
        /// `mergeJoinless`) and each assignment's (source, row) as int32 [n, k, 2]; the combine reads the
        /// rows in place. The sources stay kept until `releaseParts`.
        pub const max_parts = joinless_sources;
        pub const Parts = struct { outs: []const T, loc: T };

        fn runWideParts(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !Parts {
            // The math's unjoined prefill when it has one: the combine reads the waves' own outputs.
            try self.runWideCore(g, layer, xf, indices, n, k, comptime @hasDecl(M, "has_parts") and M.has_parts);
            const w = &self.wide;
            try w.loc.resize(self.a, 2 * n * k);
            if (comptime wide_profile.enabled) {
                const n_out = w.kept.items.len;
                const copied = try mergeJoinlessBound(G, g, self.a, &w.kept, w.pos.items, w.loc.items, &w.jl, self.joinless_bound);
                self.last_merge = .{ .outputs = @intCast(n_out), .sources = @intCast(w.kept.items.len), .copied_rows = copied };
            } else _ = try mergeJoinlessBound(G, g, self.a, &w.kept, w.pos.items, w.loc.items, &w.jl, self.joinless_bound);
            const loc = try g.hostArray(std.mem.sliceAsBytes(w.loc.items), &.{ @intCast(n), @intCast(k), 2 }, .int32);
            return .{ .outs = w.kept.items, .loc = loc };
        }

        /// Releases the outputs `runWideParts` kept (after the combines that read them are evaluated).
        pub fn releaseParts(self: *Self, g: *G) void {
            for (self.wide.kept.items) |x| g.release(x);
            self.wide.kept.clearRetainingCapacity();
        }

        /// A wide call's groups over its experts hottest first: the seed's ranks [0, sa) chunked apart from the stream's
        /// (`Wide.seed_aligned`; sa 0: one run of `max_route_ids` chunks).
        const Groups = struct {
            d: []const u16,
            sa: usize,
            seed_groups: usize,

            fn start(gs: Groups, gi: usize) usize {
                return if (gi < gs.seed_groups) gi * max_route_ids else gs.sa + (gi - gs.seed_groups) * max_route_ids;
            }

            fn of(gs: Groups, gi: usize) []const u16 {
                const s0 = gs.start(gi);
                return gs.d[s0..@min(s0 + max_route_ids, if (gi < gs.seed_groups) gs.sa else gs.d.len)];
            }

            /// The group of the expert at `fi` in the call's order (`start`'s inverse).
            fn index(gs: Groups, fi: usize) usize {
                return if (fi < gs.sa) fi / max_route_ids else gs.seed_groups + (fi - gs.sa) / max_route_ids;
            }
        };

        /// The deferred base-bank rows gathered so far as one call's waves, drained; their slots stay held. `at_seed`:
        /// P1b's call at the seed (the profile's split charges it apart from the end call's).
        fn deferredBase(self: *Self, g: *G, layer: u32, act: T, comptime parts: bool, at_seed: bool) !void {
            const a = self.a;
            const w = &self.wide;
            if (w.def_slot.items.len == 0) return;
            wide_profile.setMode(if (at_seed) .base_seed else .base_end);
            defer wide_profile.setMode(.stream);
            const kb = w.kept.items.len;
            const b = self.banks[layer][@backingInt(BankKind.base)].?;
            const rows_: quant.PrefillRows = .{ .slot = w.def_slot.items, .act_row = w.def_act.items };
            if (parts) {
                w.wave_pos.clearRetainingCapacity();
                try self.math.prefillParts(g, layer, act, rows_, b, a, &w.kept, &w.wave_pos);
                for (w.wave_pos.items) |p| try w.pos.append(a, w.def_pos.items[p]);
            } else {
                try w.kept.ensureUnusedCapacity(a, 1);
                w.kept.appendAssumeCapacity(try self.math.prefill(g, layer, act, rows_, b));
                try w.pos.appendSlice(a, w.def_pos.items);
            }
            try self.math.finishPrefill(g);
            const td = wide_profile.now();
            try g.evalAll(w.kept.items[kb..]);
            wide_profile.charge(.drain, td);
            w.def_slot.clearRetainingCapacity();
            w.def_act.clearRetainingCapacity();
            w.def_pos.clearRetainingCapacity();
        }

        /// `Wide.route_cut`: the call's rows bucketed by group (`bucketRowsByGroup`) into `inv` [n_ids] (the join's scratch,
        /// unused until the call's groups are done: no new per-row memory), group `gi`'s at `grp_off[gi]..grp_off[gi + 1]`.
        /// A group's bank passes then read its own rows only. `counted`: `count` holds each expert's rows.
        fn groupRows(a: std.mem.Allocator, w: *WideScratch, groups: Groups, n_groups: usize, counted: bool) !void {
            try w.grp_of.resize(a, w.first.items.len);
            for (groups.d, 0..) |e, i| w.grp_of.items[e] = @intCast(groups.index(i));
            try w.grp_off.resize(a, n_groups + 1);
            try w.inv.resize(a, w.ids.items.len);
            bucketRowsByGroup(w.ids.items, groups.d, w.grp_of.items, if (counted) w.count.items else null, w.grp_off.items, w.inv.items);
        }

        /// The construction's route (`Wide.route_cut`), once per call: the core at its comptime instance.
        fn runWideCore(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32, comptime parts: bool) !void {
            return switch (self.wide_route.route_cut) {
                inline else => |cut| self.runWideCoreAt(g, layer, xf, indices, n, k, parts, cut),
            };
        }

        fn runWideCoreAt(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32, comptime parts: bool, comptime cut: bool) !void {
            const a = self.a;
            const w = &self.wide;
            wide_profile.countWideCall();
            const n_ids = n * k;
            // The wide lane takes prefill-width calls only (the prefill texts bind small inputs as
            // `constant`; a decode-width call is the decode lane's).
            if (comptime !@hasDecl(M, "prefill_min_tokens")) std.debug.assert(n_ids >= wide_min_ids);
            const feed = self.wide_route.seed;
            const hot_first = self.wide_route.hot_first;
            const depth: usize = self.wide_route.depth;
            const cold: u32 = self.wide_route.cold_rows;
            const cold_chunk: usize = if (@hasDecl(M, "max_decode_rows")) M.max_decode_rows else max_route_ids;
            try w.ids.resize(a, n_ids);
            var tp = wide_profile.now();
            _ = try g.hostIds(indices, w.ids.items);
            wide_profile.charge(.barrier, tp);
            // A0 (profile builds): the prompt's last rows' experts.
            if (comptime first_cycle.enabled) first_cycle.tailRecord(layer, w.ids.items, n, k);
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            w.distinct.clearRetainingCapacity();
            const counted = feed or hot_first or cold > 0;
            if (comptime cut) {
                // The call's experts in first appearance and (the feed's order, the cold split) their rows: one pass.
                try w.distinct.ensureTotalCapacity(a, self.n_experts);
                if (counted) {
                    try w.count.resize(a, self.n_experts);
                    @memset(w.count.items, 0);
                    for (w.ids.items) |e| {
                        if (w.first.items[e] < 0) {
                            w.first.items[e] = @intCast(w.distinct.items.len);
                            w.distinct.appendAssumeCapacity(e);
                        }
                        w.count.items[e] += 1;
                    }
                } else for (w.ids.items) |e| if (w.first.items[e] < 0) {
                    w.first.items[e] = @intCast(w.distinct.items.len);
                    w.distinct.appendAssumeCapacity(e);
                };
            } else {
                for (w.ids.items) |e| if (w.first.items[e] < 0) {
                    w.first.items[e] = @intCast(w.distinct.items.len);
                    try w.distinct.append(a, e);
                };
                if (counted) {
                    try w.count.resize(a, self.n_experts);
                    @memset(w.count.items, 0);
                    for (w.ids.items) |e| w.count.items[e] += 1;
                }
            }
            // The call's residency seed (the seed route), then its experts hottest first (the order route).
            if (feed) {
                // P1: the layer's read-ahead lands before its seed (which protects the ones it chooses).
                if (comptime @hasDecl(S, "awaitReadAhead")) if (self.wide_route.read_ahead) try self.source.awaitReadAhead(layer);
                if (comptime @hasDecl(S, "seedPrefill")) try self.source.seedPrefill(layer, w.ids.items) else unreachable;
            }
            if (hot_first) {
                // A source with its own hot order (MiMo's: the P7 read schedules take the call's groups from it) orders the call.
                if (comptime @hasDecl(S, "hotOrder")) self.source.hotOrder(layer, w.distinct.items, w.count.items) else std.sort.pdq(u16, w.distinct.items, @as([]const u32, w.count.items), struct {
                    fn lt(c: []const u32, x: u16, y: u16) bool {
                        return if (c[x] != c[y]) c[x] > c[y] else x < y;
                    }
                }.lt);
                for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            }
            const act = if (g.dtypeOf(xf) == .bfloat16) xf else try g.astype(xf, .bfloat16);
            w.pos.clearRetainingCapacity();
            w.kept.clearRetainingCapacity();
            const defer_base = self.wide_route.defer_base;
            w.def_slot.clearRetainingCapacity();
            w.def_act.clearRetainingCapacity();
            w.def_pos.clearRetainingCapacity();
            const n_distinct = w.distinct.items.len;
            // Seed-aligned: the seed's ranks [0, sa) in groups of their own, the stream's after them (sa 0: one run).
            const aligned = self.wide_route.seed_aligned;
            const sa: usize = if (comptime @hasDecl(S, "seedRanks")) (if (aligned) @min(self.source.seedRanks(layer), n_distinct) else 0) else 0;
            const groups: Groups = .{ .d = w.distinct.items, .sa = sa, .seed_groups = (sa + max_route_ids - 1) / max_route_ids };
            const n_groups = groups.seed_groups + (n_distinct - sa + max_route_ids - 1) / max_route_ids;
            if (comptime cut) try groupRows(a, w, groups, n_groups, counted);
            // P1b: the seed's base call is due after its last group (aligned: the last seed group; else the first group
            // with a transient row).
            var base_due = self.wide_route.base_at_seed;
            // The live groups' calls, by group index mod `depth` (read ahead up to `depth` groups).
            var calls: [expert_stream.max_wide_depth]?*S.Call = @splat(null);
            errdefer {
                for (&calls) |*c| if (c.*) |cl| {
                    self.source.release(cl);
                    c.* = null;
                };
                for (w.kept.items) |x| g.release(x);
                w.kept.clearRetainingCapacity();
                if (comptime @hasDecl(S, "holdBase")) if (defer_base) self.source.releaseHeld();
            }
            tp = wide_profile.now();
            for (0..@min(depth, n_groups)) |gi| calls[gi % depth] = try self.source.route(layer, groups.of(gi), &.{});
            wide_profile.charge(.route, tp);
            for (0..n_groups) |gi| {
                const start = groups.start(gi);
                const group = groups.of(gi);
                const call = calls[gi % depth].?;
                const sv = self.source.served(call);
                tp = wide_profile.now();
                const ready: ?u64 = if (comptime wide_event_gates) blk: {
                    // Register only the group whose consumer is encoded next. Its final down value
                    // proves every gate/up and down read in the group complete.
                    const gates = (try self.source.gate(call)) orelse break :blk null;
                    break :blk gates.down_first + gates.n_parts - 1;
                } else blk: {
                    for (0..sv.n_parts) |p| {
                        try self.source.waitGu(call, @intCast(p));
                        try self.source.waitDown(call, @intCast(p));
                    }
                    break :blk null;
                };
                wide_profile.charge(.read_wait, tp);
                const k0 = w.kept.items.len;
                var streamed = false;
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    const kb = w.kept.items.len;
                    w.slot.clearRetainingCapacity();
                    w.act_row.clearRetainingCapacity();
                    w.cold_slot.clearRetainingCapacity();
                    w.cold_act.clearRetainingCapacity();
                    w.cold_pos.clearRetainingCapacity();
                    // The cut: the group's own rows, in row order (`groupRows`), each bank's rows as a pass over the whole
                    // call meets them; the control: every row of the call, the group's kept.
                    const passed: []const u32 = if (comptime cut) w.inv.items[w.grp_off.items[gi]..w.grp_off.items[gi + 1]] else &.{};
                    const n_pass = if (comptime cut) passed.len else w.ids.items.len;
                    for (0..n_pass) |pi| {
                        const row: usize = if (comptime cut) passed[pi] else pi;
                        const e = w.ids.items[row];
                        const fi: usize = @intCast(w.first.items[e]);
                        if (comptime !cut) if (fi < start or fi >= start + group.len) continue;
                        const ref = sv.refs[fi - start];
                        if (ref.bank != kind) continue;
                        const act_row: u32 = @intCast(row / @as(usize, k));
                        if (defer_base and kind == .base) {
                            try w.def_slot.append(a, ref.row);
                            try w.def_act.append(a, act_row);
                            try w.def_pos.append(a, @intCast(row));
                            continue;
                        }
                        if (cold > 0 and w.count.items[e] <= cold) {
                            try w.cold_slot.append(a, ref.row);
                            try w.cold_act.append(a, act_row);
                            try w.cold_pos.append(a, @intCast(row));
                            continue;
                        }
                        try w.slot.append(a, ref.row);
                        try w.act_row.append(a, act_row);
                        if (parts) try w.call_pos.append(a, @intCast(row)) else try w.pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0 and w.cold_slot.items.len == 0) continue;
                    streamed = streamed or kind == .transient;
                    var b = self.banks[layer][@backingInt(kind)].?;
                    if (comptime wide_event_gates) if (ready) |value| {
                        b = .{
                            .gate = try self.waitProj(g, b.gate, value, &.{}),
                            .up = try self.waitProj(g, b.up, value, &.{}),
                            .down = try self.waitProj(g, b.down, value, &.{}),
                        };
                    };
                    const hot = w.slot.items.len > 0;
                    if (hot) {
                        if (parts) {
                            // The waves' outputs as they are; each wave-ordered row's routed position.
                            w.wave_pos.clearRetainingCapacity();
                            try self.math.prefillParts(g, layer, act, .{ .slot = w.slot.items, .act_row = w.act_row.items }, b, a, &w.kept, &w.wave_pos);
                            for (w.wave_pos.items) |p| try w.pos.append(a, w.call_pos.items[p]);
                            w.call_pos.clearRetainingCapacity();
                        } else {
                            try w.kept.ensureUnusedCapacity(a, 1);
                            const y = try self.math.prefill(g, layer, act, .{ .slot = w.slot.items, .act_row = w.act_row.items }, b);
                            w.kept.appendAssumeCapacity(y);
                        }
                    }
                    // Cold rows: the decode GEMV over their slots, encoded after the bank's wide waves
                    // (the act-row index is built right before its take).
                    var c0: usize = 0;
                    while (c0 < w.cold_slot.items.len) : (c0 += cold_chunk) {
                        const c1 = @min(c0 + cold_chunk, w.cold_slot.items.len);
                        const m: c_int = @intCast(c1 - c0);
                        const ar = try g.hostArray(std.mem.sliceAsBytes(w.cold_act.items[c0..c1]), &.{m}, .uint32);
                        const xs = try g.take(act, ar, 0);
                        const sid = try g.hostArray(std.mem.sliceAsBytes(w.cold_slot.items[c0..c1]), &.{m}, .uint32);
                        const h = try self.math.gateUp(g, xs, sid, b.gate, b.up);
                        try w.kept.ensureUnusedCapacity(a, 1);
                        w.kept.appendAssumeCapacity(g.keep(try self.math.down(g, h, sid, b.down)));
                        try w.pos.appendSlice(a, w.cold_pos.items[c0..c1]);
                    }
                    if (!feed) {
                        if (hot) try self.math.finishPrefill(g);
                        tp = wide_profile.now();
                        try g.evalAll(w.kept.items[kb..]);
                        wide_profile.charge(.drain, tp);
                    }
                }
                // Feed: the group's banks queued back to back, one drain before its slots go back.
                if (feed) {
                    try self.math.finishPrefill(g);
                    tp = wide_profile.now();
                    try g.evalAll(w.kept.items[k0..]);
                    wide_profile.charge(.drain, tp);
                }
                tp = wide_profile.now();
                // The group's base-bank slots stay pinned and held for the deferred call past its release.
                if (comptime @hasDecl(S, "holdBase")) if (defer_base) try self.source.holdBase(call);
                self.source.release(call);
                calls[gi % depth] = null;
                if (gi + depth < n_groups) calls[gi % depth] = try self.source.route(layer, groups.of(gi + depth), &.{});
                wide_profile.charge(.route, tp);
                // P1b: the seed has landed (its groups read, the next routes' reads posted): its base call drains now.
                if (base_due and (if (aligned) gi + 1 == groups.seed_groups else streamed)) {
                    base_due = false;
                    try self.deferredBase(g, layer, act, parts, true);
                }
            }
            // The deferred base-bank rows: one call's waves over every group's (with P1b, the later ones'), then the
            // slots let go.
            if (comptime @hasDecl(S, "holdBase")) if (defer_base) {
                try self.deferredBase(g, layer, act, parts, false);
                self.source.releaseHeld();
            };
        }
    };
}

// ── Tests ──

const testing = std.testing;
const TraceOps = ops.TraceOps;

comptime {
    assertSource(StreamSource);
    assertSource(FakeSource);
}

/// Scripted host reads for the trace backend: the routed ids of each call.
const Script = struct {
    calls: []const []const u16,
    next: usize = 0,

    fn values(self: *Script) TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }

    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *Script = @ptrCast(@alignCast(ctx));
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

fn kindsOf(log: []const FakeSource.Event, buf: []u8) []const u8 {
    for (log, buf[0..log.len]) |e, *c| c.* = switch (e.kind) {
        .route => 'R',
        .wait_gu => 'g',
        .wait_down => 'd',
        .release => 'r',
        .flush => 'f',
        .grow => 'G',
        .gate => 'E',
        .read_ahead => 'A',
        .await_read_ahead => 'W',
    };
    return buf[0..log.len];
}

/// Distinct banks among the positions of wave `w` (the groups the wave runs).
fn groupsOf(sv: Served, w: u8) usize {
    var seen: [n_banks]bool = @splat(false);
    var n: usize = 0;
    for (sv.waves, sv.refs) |wv, ref| {
        if (wv != w or seen[@backingInt(ref.bank)]) continue;
        seen[@backingInt(ref.bank)] = true;
        n += 1;
    }
    return n;
}

/// Trace nodes of kind `op` between consecutive log events: [0] before the
/// first event, [j] between events j - 1 and j, [len] after the last.
fn opsBetween(g: *const TraceOps, log: []const FakeSource.Event, first: usize, op: ops.Op, out: []usize) void {
    @memset(out, 0);
    for (g.nodes.items[first..], first..) |nd, i| if (nd.op == op) {
        const j = for (log, 0..) |e, jj| {
            if (e.at > i) break jj;
        } else log.len;
        out[j] += 1;
    };
}

const TestConfig = struct { hidden_size: u32, moe_intermediate_size: u32, n_layers: u32, n_experts_per_tok: u32 = 6, swiglu_limit: f64 = 10.0, n_routed_experts: u32 = undefined };

fn testConfig(hidden: u32, inter: u32, n_layers: u8) TestConfig {
    return .{ .hidden_size = hidden, .moe_intermediate_size = inter, .n_layers = n_layers };
}

test "dsv41 experts: the fake and the stream adapter are expert sources" {
    comptime assertSource(StreamSource);
    comptime assertSource(FakeSource);
    try testing.expectEqual(@as(usize, expert_stream.max_wide_depth + 1), StreamSource.n_routes);
}

test "dsv41 experts: a decode call runs the residents, then each part's gate/up after waitGu and its down after waitDown" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2); // widths an EXL3 bank has (t128 blocks of 128)
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = Experts(TraceOps, FakeSource, Chain);
    var ex = try Ex.init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try testing.expect(ex.banks[1][@backingInt(BankKind.ext)] == null);
    try ex.grow(&g, &.{ 8, 8 });
    try testing.expect(ex.banks[1][@backingInt(BankKind.ext)] != null);
    src.log.clearRetainingCapacity();
    src.trace = &g;

    var script: Script = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
    g.host_values = script.values();
    const xf = try g.input(&.{ 2, 256 }, .bfloat16);
    const idx = try g.input(&.{ 2, 3 }, .int32);

    // Call 1 (layer 0, 2 rows x top-3): six misses, two parts of three in expert order.
    const first = g.nodes.items.len;
    const y = try ex.at(0).routed(&g, xf, idx);
    try testing.expect(g.shapeOf(y).eql(ops.Shape.of(&.{ 2, 3, 256 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(y));
    var kb: [64]u8 = undefined;
    try testing.expectEqualStrings("Rgdgdr", kindsOf(src.log.items, &kb));
    const sv = src.served(&src.calls[0]);
    try testing.expectEqual(@as(u32, 2), sv.n_parts);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 2, 2, 2 }, sv.waves);
    const g1 = groupsOf(sv, 1);
    const g2 = groupsOf(sv, 2);
    // Kernels per interval [<R, R..g0, g0..d0, d0..g1, g1..d1, d1..r, >r]: none
    // before a publish, gate + up per group after waitGu, down per group after waitDown.
    var k: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .kernel, &k);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 2 * g1, g1, 2 * g2, g2, 0 }, &k);
    // The routing barrier comes before the route; each wave starts on the GPU.
    var hr: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .host_read, &hr);
    try testing.expectEqualSlices(usize, &.{ 1, 0, 0, 0, 0, 0, 0 }, &hr);
    var ae: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .async_eval, &ae);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1, 1, 1, 0 }, &ae);
    // Two t128 (reshape, hadamard, reshape) around each of the three GEMVs of a group.
    var hd: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .hadamard, &hd);
    try testing.expectEqual(4 * g1 + 2 * g1 + 4 * g2 + 2 * g2, hd[2] + hd[3] + hd[4] + hd[5]);

    // Call 2 (same layer): the released call is flushed by the route; 1, 2, 3
    // are resident (wave 0, at once), 7 and 9 load in one part.
    const mark = src.log.items.len;
    const second = g.nodes.items.len;
    _ = try ex.at(0).routed(&g, xf, idx);
    try testing.expectEqualStrings("fRgdr", kindsOf(src.log.items[mark..], &kb));
    const sv2 = src.served(&src.calls[0]);
    try testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 1, 0 }, sv2.waves);
    const h0 = groupsOf(sv2, 0);
    const h1 = groupsOf(sv2, 1);
    var k2: [6]usize = undefined;
    opsBetween(&g, src.log.items[mark..], second, .kernel, &k2);
    // [<f, f..R, R..g0 (the residents' gate/up + down), g0..d0, d0..r, >r]
    try testing.expectEqualSlices(usize, &.{ 0, 0, 3 * h0, 2 * h1, h1, 0 }, &k2);
    const st = src.stats();
    try testing.expectEqual(@as(u64, 2), st.route_calls);
    try testing.expectEqual(@as(u64, 3), st.expert_cache_hits);
    try testing.expectEqual(@as(u64, 8), st.expert_cache_misses);
    try ex.flush();
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
    try testing.expectEqual(FakeSource.Event.Kind.flush, src.log.items[src.log.items.len - 1].kind);
}

test "dsv41 experts: a wider call is the prefill lane's (not ported), refused before any route" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 1);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{4} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    var ex = try Experts(TraceOps, FakeSource, Chain).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    const xf = try g.input(&.{ 9, 256 }, .bfloat16);
    const idx = try g.input(&.{ 9, 6 }, .int32);
    try testing.expectError(error.PrefillLaneNotPorted, ex.at(0).routed(&g, xf, idx));
    try testing.expectEqual(@as(usize, 0), src.log.items.len);
}

/// The wide lane's host tests: a registry from the embedded manifest (no MLX).
fn hostRegistry() !xk.Registry {
    var diag: xk.Diag = .{};
    return xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("dsv41 experts: {s}\n", .{diag.message()});
        return e;
    };
}

/// A wide-lane route that records every call: its rows, the bank it was
/// handed and, from the fake source's live route, the served slots of the
/// group (so the test can derive what each call should have been).
const RecRoute = struct {
    a: std.mem.Allocator,
    calls: std.ArrayList(Rec) = .empty,
    finishes: u32 = 0,

    /// The source whose live route each call snapshots.
    var source: ?*FakeSource = null;

    const Rec = struct { slot: []u32, act_row: []u32, bank_code: u32, route: u64, refs: []SlotRef, at: usize, node: usize, act_dtype: ops.Dtype };

    pub fn init(a: std.mem.Allocator) RecRoute {
        return .{ .a = a };
    }

    pub fn deinit(self: *RecRoute, _: *TraceOps) void {
        for (self.calls.items) |r| {
            self.a.free(r.slot);
            self.a.free(r.act_row);
            self.a.free(r.refs);
        }
        self.calls.deinit(self.a);
    }

    pub fn call(self: *RecRoute, g: *TraceOps, act: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const src = source.?;
        // A deferred base-bank call runs after its groups' routes are released: no refs to snapshot.
        const live_refs: []const SlotRef = for (&src.calls) |*c| {
            if (c.state == .live) break c.refs[0..c.plan.n_ids];
        } else &.{};
        try self.calls.append(self.a, .{
            .slot = try self.a.dupe(u32, rows.slot),
            .act_row = try self.a.dupe(u32, rows.act_row.?),
            .bank_code = bank.gate.code,
            .route = src.counters.route_calls,
            .refs = try self.a.dupe(SlotRef, live_refs),
            .at = src.log.items.len,
            .node = g.nodes.items.len,
            .act_dtype = g.dtypeOf(act),
        });
        return g.input(&.{ @intCast(rows.slot.len), g.shapeOf(act).d[1] }, .float32);
    }

    pub fn finish(self: *RecRoute, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

test "mimo sdk2 experts: bank bindings derive from the math's native projection" {
    const NativeMath = struct {
        pub const Projection = quant.GatherQmm.Arrays;
    };
    const Ex = ExpertsWith(TraceOps, FakeSource, NativeMath, .{ .prefill = true });
    const Native = quant.GatherQmm.Arrays(u32);
    try testing.expect(Ex.Arrays == quant.BankArrays(Native));
    try testing.expect(@hasField(@FieldType(Ex.Arrays, "gate"), "w"));
    try testing.expect(!@hasField(@FieldType(Ex.Arrays, "gate"), "code"));
    // Math without a projection override keeps DeepSeek's existing signature.
    const Default = ExpertsWith(TraceOps, FakeSource, TraceMath, .{ .prefill = true });
    try testing.expect(Default.Arrays == BankArraysOf(u32));
}

test "mimo sdk2 experts: event waits register the native MXFP4 pair and the EXL3 triple in projection order" {
    const a = testing.allocator;
    var source = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{0} });
    defer source.deinit();
    const NativeWaitMath = struct {
        pub const Projection = quant.GatherQmm.Arrays;
        hidden: c_int,
        inter: c_int,
        pub fn EventProjection(comptime T: type) type {
            return struct {
                pub const count: usize = 2;
                pub fn inputs(p: Projection(T)) [count]T {
                    return .{ p.w, p.scales.? };
                }
                pub fn fromWaited(out: [count]T) Projection(T) {
                    return .{ .w = out[0], .scales = out[1], .biases = null };
                }
            };
        }
        pub fn gateUp(_: *const @This(), _: *TraceOps, _: u32, _: u32, _: Projection(u32), _: Projection(u32)) !u32 {
            return error.UnusedEventWaitMath;
        }
        pub fn down(_: *const @This(), _: *TraceOps, _: u32, _: u32, _: Projection(u32)) !u32 {
            return error.UnusedEventWaitMath;
        }
    };
    inline for (.{ NativeWaitMath, TraceMath }) |WaitMath| {
        const ProjectionType = if (@hasDecl(WaitMath, "Projection")) WaitMath.Projection else ProjOf;
        const Ex = ExpertsWith(TraceOps, FakeSource, WaitMath, .{ .gated = true });
        var g = TraceOps.init(a);
        defer g.deinit();
        var ex: Ex = .{ .a = a, .source = &source, .math = .{ .hidden = 256, .inter = 128 }, .hidden = 256, .n_experts = 16, .banks = &.{}, .g = &g };
        const dtypes = if (comptime WaitMath == NativeWaitMath) [_]ops.Dtype{ .uint32, .uint8 } else [_]ops.Dtype{ .int16, .float16, .float16 };
        var xs: [dtypes.len]u32 = undefined;
        for (&xs, dtypes, 0..) |*x, dtype, i| x.* = try g.input(&.{ 2, @intCast(8 + i) }, dtype);
        const projection: ProjectionType(u32) = if (comptime WaitMath == NativeWaitMath)
            .{ .w = xs[0], .scales = xs[1], .biases = null }
        else
            .{ .code = xs[0], .rout = xs[1], .rin = xs[2] };
        const deps = [_]u32{ try g.input(&.{1}, .float32), try g.input(&.{2}, .float32) };
        const mark = g.mark();
        const waited = try ex.waitProj(&g, projection, 17, &deps);
        const aliases = if (comptime WaitMath == NativeWaitMath)
            [_]u32{ waited.w, waited.scales.? }
        else
            [_]u32{ waited.code, waited.rout, waited.rin };
        if (comptime WaitMath == NativeWaitMath) try testing.expect(waited.biases == null);
        try testing.expectEqual(mark.n + xs.len, g.nodes.items.len);
        try testing.expectEqual(xs.len, g.waits.items.len);
        for (xs, aliases, 0..) |original, alias, i| {
            try testing.expectEqual(@as(u32, @intCast(mark.n + i)), alias);
            try testing.expect(alias != original);
            try testing.expectEqual(ops.Op.event_wait, g.nodes.items[alias].op);
            try testing.expectEqual(g.dtypeOf(original), g.dtypeOf(alias));
            try testing.expect(g.shapeOf(original).eql(g.shapeOf(alias)));
            try testing.expectEqual(TraceOps.Wait{ .value = 17, .n_deps = deps.len }, g.waits.items[i]);
        }
        // Registered aliases belong to the graph scope; the source arrays predate it.
        g.resetTo(mark);
        try testing.expectEqual(TraceOps.Freed{ .from = @intCast(mark.n), .to = @intCast(mark.n + xs.len) }, g.freed.items[0]);
        for (xs) |x| try testing.expect(x < mark.n);
    }
}

/// A test-only profile hook proves the optional math hook receives the shared
/// schedule's charges without reading clocks or initializing MLX.
const SmallPrefillProfile = struct {
    pub const enabled = true;
    pub const Mode = prof.Mode;
    pub var wide_calls: u32 = 0;
    pub var charges: u32 = 0;
    pub fn now() u64 {
        return 0;
    }
    pub fn charge(_: sdk_ext.profile.PrefillBucket, _: u64) void {
        charges += 1;
    }
    pub fn setMode(_: Mode) void {}
    pub fn countWideCall() void {
        wide_calls += 1;
    }
};

const SmallPrefillMath = struct {
    const Chain = EagerChain(TraceOps, TraceGemv);
    pub const decode_forward_rows: u32 = 1;
    pub const prefill_min_tokens: u32 = 2;
    pub const wide_event_gates = true;
    pub const Profile = SmallPrefillProfile;
    d: Chain,
    route: *RecRoute,
    pub fn gateUp(self: *const SmallPrefillMath, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        return self.d.gateUp(g, x, ids, gate, up);
    }
    pub fn down(self: *const SmallPrefillMath, g: *TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.d.down(g, h, ids, d);
    }
    pub fn prefill(self: *const SmallPrefillMath, g: *TraceOps, _: u32, x: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const mark = g.mark();
        const y = g.keep(try self.route.call(g, x, rows, bank));
        g.resetTo(mark);
        return y;
    }
    pub fn finishPrefill(self: *const SmallPrefillMath, g: *TraceOps) !void {
        return self.route.finish(g);
    }
};

test "mimo sdk2 experts: top8 one-token decode and two-token prefill preserve assignments with generic config" {
    const a = testing.allocator;
    // This is deliberately a different architecture's minimal config type.
    const c = .{ .hidden_size = @as(u32, 256), .n_layers = @as(u32, 1), .n_routed_experts = @as(u32, 32), .n_experts_per_tok = @as(u32, 8) };
    const decode_config = testConfig(256, 128, 1);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 32, .rows = &.{8}, .records_per_part = 2 });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    src.trace = &g;
    RecRoute.source = &src;
    defer RecRoute.source = null;
    var rec = RecRoute.init(a);
    defer rec.deinit(&g);
    SmallPrefillProfile.wide_calls = 0;
    SmallPrefillProfile.charges = 0;
    const Ex = ExpertsWith(TraceOps, FakeSource, SmallPrefillMath, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = SmallPrefillMath.Chain.init(.{}, &decode_config), .route = &rec }, &c);
    defer ex.deinit();
    const prefill_ids = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11 };
    var script: Script = .{ .calls = &.{ &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &prefill_ids } };
    g.host_values = script.values();
    const decode = try ex.at(0).routed(&g, try g.input(&.{ 1, 256 }, .bfloat16), try g.input(&.{ 1, 8 }, .int32));
    try testing.expect(g.shapeOf(decode).eql(ops.Shape.of(&.{ 1, 8, 256 })));
    try testing.expectEqual(@as(usize, 0), rec.calls.items.len);
    try ex.flush();
    const prefill = try ex.at(0).routed(&g, try g.input(&.{ 2, 256 }, .bfloat16), try g.input(&.{ 2, 8 }, .int32));
    try testing.expect(g.shapeOf(prefill).eql(ops.Shape.of(&.{ 2, 8, 256 })));
    try testing.expect(rec.calls.items.len > 0);
    var seen: [2]u32 = @splat(0);
    for (rec.calls.items) |call| {
        try testing.expectEqual(call.slot.len, call.act_row.len);
        for (call.act_row) |token| seen[token] += 1;
        // Each group's rows are evaluated before release makes its slots reusable.
        const release_at = for (src.log.items[call.at..]) |event| {
            if (event.kind == .release) break event.at;
        } else return error.NoRelease;
        var evaluated = false;
        for (g.evals.items) |node| evaluated = evaluated or (node >= call.node and node <= release_at);
        try testing.expect(evaluated);
    }
    try testing.expectEqualSlices(u32, &.{ 8, 8 }, &seen);
    try testing.expectEqual(@as(u32, 1), SmallPrefillProfile.wide_calls);
    try testing.expect(SmallPrefillProfile.charges > 0);
    try ex.flush();
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
}

test "mimo sdk2 experts: wide gates wait for the current group's complete reads and keep aliases through its drain" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 64;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const PlainMath = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    inline for (.{ false, true }) |wide_gated| {
        const Math = if (wide_gated) SmallPrefillMath else PlainMath;
        const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true, .gated = true });
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 64, .rows = &.{64} });
        defer src.deinit();
        var g = TraceOps.init(a);
        defer g.deinit();
        src.trace = &g;
        RecRoute.source = &src;
        defer RecRoute.source = null;
        var rrs = [_]RecRoute{RecRoute.init(a)};
        defer rrs[0].deinit(&g);
        const math: Math = if (wide_gated)
            .{ .d = Chain.init(.{}, &c), .route = &rrs[0] }
        else
            .{ .d = Chain.init(.{}, &c), .routes = &rrs };
        var ex = try Ex.init(a, &g, &src, math, &c);
        defer ex.deinit();
        const n = 12;
        const k = 6;
        var ids: [n * k]u16 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast(i % 64);
        var script: Script = .{ .calls = &.{ &ids, &ids } };
        g.host_values = script.values();
        const xf = try g.input(&.{ n, 256 }, .bfloat16);
        const indices = try g.input(&.{ n, k }, .int32);
        _ = try ex.at(0).routed(&g, xf, indices);
        try testing.expectEqual(@as(usize, 2), rrs[0].calls.items.len);
        var kinds_buf: [64]u8 = undefined;
        const kinds = kindsOf(src.log.items, &kinds_buf);
        try testing.expectEqual(@as(usize, if (wide_gated) 2 else 0), std.mem.count(u8, kinds, "E"));
        try testing.expectEqual(@as(usize, 2), std.mem.count(u8, kinds, "r"));
        if (wide_gated) {
            try testing.expectEqual(@as(usize, 0), std.mem.count(u8, kinds, "g"));
            try testing.expectEqual(@as(usize, 0), std.mem.count(u8, kinds, "d"));
            try testing.expectEqual(@as(usize, 18), g.waits.items.len);
            var lo: u64 = 0;
            var group: usize = 0;
            for (src.log.items) |event| if (event.kind == .gate) {
                const ready = lo + 1 + event.part;
                const rec = rrs[0].calls.items[group];
                // Registration precedes this group's aliases and consumer, never a future group's wait.
                try testing.expectEqual(@as(u32, @intCast(event.at)), rec.bank_code);
                try testing.expect(rec.bank_code < rec.node);
                for (g.waits.items[group * 9 ..][0..9]) |wait| try testing.expectEqual(TraceOps.Wait{ .value = ready, .n_deps = 0 }, wait);
                // Prefill's inner scope releases its result graph while the bank's registered aliases stay outside it.
                try testing.expectEqual(@as(u32, @intCast(rec.node)), g.freed.items[group].from);
                try testing.expect(rec.bank_code + 9 <= g.freed.items[group].from);
                lo = ready;
                group += 1;
            };
            try testing.expectEqual(@as(usize, 2), group);
        } else {
            // DeepSeek's explicit wide route still uses host waits even with decode gates installed.
            try testing.expect(std.mem.count(u8, kinds, "g") > 0);
            try testing.expectEqual(@as(usize, 0), g.waits.items.len);
        }
        for (rrs[0].calls.items) |rec| {
            const released = for (src.log.items[rec.at..]) |event| {
                if (event.kind == .release) break event.at;
            } else return error.NoRelease;
            var drained = false;
            for (g.evals.items) |at| drained = drained or (at > rec.node and at <= released);
            try testing.expect(drained);
        }
        // Repeating the same groups serves only residents: no external event producer or alias.
        const waits_before = g.waits.items.len;
        src.log.clearRetainingCapacity();
        _ = try ex.at(0).routed(&g, xf, indices);
        try testing.expectEqual(waits_before, g.waits.items.len);
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, kindsOf(src.log.items, &kinds_buf), "E"));
        const original = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        for (rrs[0].calls.items[2..]) |rec| try testing.expectEqual(original, rec.bank_code);
        try ex.flush();
        try testing.expectEqual(@as(usize, 0), src.liveCalls());
    }
}

test "dsv41 experts: a wide call routes its experts in groups and runs each bank's rows through the wide route" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 64, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    src.trace = &g;
    RecRoute.source = &src;
    defer RecRoute.source = null;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    var rrs = [_]RecRoute{RecRoute.init(a)};
    defer rrs[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c);
    defer ex.deinit();
    // 20 tokens x top-6 over 64 experts: 120 rows, every expert id distinct within a token.
    const n = 20;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast((7 * (i / k) + 11 * (i % k)) % 64);
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    // An f32 stream reaches the wide route rounded to bf16 (the DIG kernels' activations).
    const out = try ex.at(0).routed(&g, try g.input(&.{ n, 256 }, .float32), try g.input(&.{ n, k }, .int32));
    try testing.expect(g.shapeOf(out).eql(ops.Shape.of(&.{ n, k, 256 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(out));

    // The call's distinct experts, in first appearance, routed in groups of max_route_ids.
    var first: [64]i32 = @splat(-1);
    var distinct: std.ArrayList(u16) = .empty;
    defer distinct.deinit(a);
    for (ids) |e| if (first[e] < 0) {
        first[e] = @intCast(distinct.items.len);
        try distinct.append(a, e);
    };
    const n_groups = (distinct.items.len + max_route_ids - 1) / max_route_ids;
    try testing.expectEqual(@as(u64, n_groups), src.counters.route_calls);
    // The source protocol: per group route, every part's waits, release (the next route flushes it).
    var kb: [64]u8 = undefined;
    const kinds = kindsOf(src.log.items, &kb);
    try testing.expectEqual(@as(usize, n_groups), std.mem.count(u8, kinds, "R"));
    try testing.expectEqual(@as(usize, n_groups), std.mem.count(u8, kinds, "r"));
    try testing.expectEqual(std.mem.count(u8, kinds, "g"), std.mem.count(u8, kinds, "d"));
    // The default wide route (no feed, depth 1): one group live at a time, and one finish + eval
    // per bank call (the schedule before the wide routes existed).
    var open = false;
    for (kinds) |kd| switch (kd) {
        'R' => {
            try testing.expect(!open);
            open = true;
        },
        'r' => {
            try testing.expect(open);
            open = false;
        },
        else => {},
    };
    try testing.expect(!open);
    try testing.expectEqual(rrs[0].calls.items.len, g.evals.items.len);

    // Each recorded call is exactly its group's rows in one bank, in routed order, read
    // from the slot the source serves the expert in, after the group's waits and before
    // its release; together they cover every row once.
    const rr = &rrs[0];
    try testing.expect(rr.calls.items.len >= n_groups);
    try testing.expectEqual(@as(u32, @intCast(rr.calls.items.len)), rr.finishes);
    var rows_seen: usize = 0;
    for (rr.calls.items) |rec| {
        try testing.expectEqual(ops.Dtype.bfloat16, rec.act_dtype);
        const gidx: usize = @intCast(rec.route - 1);
        const kind: BankKind = for ([_]BankKind{ .base, .ext, .transient }) |kd| {
            if (ex.banks[0][@backingInt(kd)]) |b| if (b.gate.code == rec.bank_code) break kd;
        } else return error.UnknownBank;
        var want_slot: std.ArrayList(u32) = .empty;
        defer want_slot.deinit(a);
        var want_act: std.ArrayList(u32) = .empty;
        defer want_act.deinit(a);
        for (ids, 0..) |e, row| {
            const gi: usize = @intCast(first[e]);
            if (gi / max_route_ids != gidx) continue;
            const ref = rec.refs[gi - gidx * max_route_ids];
            if (ref.bank != kind) continue;
            try want_slot.append(a, ref.row);
            try want_act.append(a, @intCast(row / k));
        }
        try testing.expectEqualSlices(u32, want_slot.items, rec.slot);
        try testing.expectEqualSlices(u32, want_act.items, rec.act_row);
        // after the group's route and waits, before its release
        try testing.expect(kinds[rec.at - 1] == 'd' or kinds[rec.at - 1] == 'R');
        try testing.expect(rec.at < kinds.len);
        // evaluated before the group is released (the next route may refill its slots)
        const release_at = for (src.log.items[rec.at..]) |e| {
            if (e.kind == .release) break e.at;
        } else return error.NoRelease;
        var evaluated = false;
        for (g.evals.items) |ev| evaluated = evaluated or (ev >= rec.node and ev <= release_at);
        try testing.expect(evaluated);
        rows_seen += rec.slot.len;
    }
    try testing.expectEqual(@as(usize, n * k), rows_seen);
}

test "dsv41 experts: the deferred base-bank call builds every group's base rows once, after the groups' transient calls, the slots held until then" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 128;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    // 40 tokens x top-6 over 128 experts: three groups, 32 persistent rows (the fake source holds one
    // route window: depth 1). Split, each group loads its own persistent rows and pays a base call.
    const n = 40;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast((7 * (i / k) + 11 * (i % k)) % 128);
    var calls_total: [2]usize = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |deferred, run| {
        var reg = try hostRegistry();
        defer reg.deinit();
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 128, .rows = &.{32} });
        defer src.deinit();
        var g = TraceOps.init(a);
        defer g.deinit();
        src.trace = &g;
        RecRoute.source = &src;
        defer RecRoute.source = null;
        var rrs = [_]RecRoute{RecRoute.init(a)};
        defer rrs[0].deinit(&g);
        var ex = try Ex.initWith(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c, .{ .wide = .{ .defer_base = deferred } });
        defer ex.deinit();
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 256 }, .float32), try g.input(&.{ n, k }, .int32));
        // Every routed position placed once, every row built once.
        var seen: [n * k]bool = @splat(false);
        for (ex.wide.pos.items) |p| {
            try testing.expect(!seen[p]);
            seen[p] = true;
        }
        try testing.expectEqual(@as(usize, n * k), ex.wide.pos.items.len);
        var rows: usize = 0;
        var base_calls: usize = 0;
        for (rrs[0].calls.items, 0..) |rec, ci| {
            rows += rec.slot.len;
            const kind: BankKind = for ([_]BankKind{ .base, .ext, .transient }) |kd| {
                if (ex.banks[0][@backingInt(kd)]) |b| if (b.gate.code == rec.bank_code) break kd;
            } else return error.UnknownBank;
            if (kind != .base) continue;
            base_calls += 1;
            if (!deferred) continue;
            // The one deferred call comes last, and each of its slots still holds an expert its token
            // routed (held since its group: never evicted, never refilled).
            try testing.expectEqual(rrs[0].calls.items.len - 1, ci);
            for (rec.slot, rec.act_row) |sl, t| {
                const e = src.policies[0].slot_to_expert[sl];
                try testing.expect(std.mem.indexOfScalar(u16, ids[t * k ..][0..k], e) != null);
            }
        }
        try testing.expectEqual(@as(usize, n * k), rows);
        try testing.expect(if (deferred) base_calls == 1 else base_calls >= 2);
        try testing.expectEqual(@as(usize, 0), src.held.items.len);
        calls_total[run] = rrs[0].calls.items.len;
    }
    try testing.expect(calls_total[1] < calls_total[0]);
}

/// dump_prefill_waves.route_rows: slot j repeated counts[j] times, then
/// Fisher-Yates from the end with j = splitmix64(seed) output % (i + 1).
fn sampleRows(a: std.mem.Allocator, seed: u64, slots: []const u32, counts: []const u32) ![]u16 {
    var n: usize = 0;
    for (counts) |c| n += c;
    const rows = try a.alloc(u16, n);
    var i: usize = 0;
    for (slots, counts) |s, c| for (0..c) |_| {
        rows[i] = @intCast(s);
        i += 1;
    };
    var st = seed;
    var j = n;
    while (j > 1) {
        j -= 1;
        const r: usize = @intCast(xk.splitmix64(&st) % (j + 1));
        std.mem.swap(u16, &rows[j], &rows[r]);
    }
    return rows;
}

test "dsv41 experts: the kernels' decode GEMV launches configs prepared on the model's backend at construction" {
    var reg = try hostRegistry();
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    {
        var gemv = try xq.Gemv(TraceOps).init(&g, &reg);
        defer gemv.deinit(&g);
        try testing.expect(g.prepared_live > 0);
        // gate / up: xh f32 [rows, 5120] at slot rows ids, code i16 [cap, 320, 144, 48] -> [rows, 2304] f32
        const z = try gemv.project(&g, .gate, try g.input(&.{ 6, 5120 }, .float32), try g.input(&.{6}, .uint32), try g.input(&.{ 64, 320, 144, 48 }, .int16));
        try testing.expect(g.shapeOf(z).eql(ops.Shape.of(&.{ 6, 2304 })));
        try testing.expectEqual(@as(usize, 1), g.prepared_launches);
    }
    try testing.expectEqual(@as(usize, 0), g.prepared_live);
}

test "dsv41 experts: the joined outputs are put back in routed order" {
    // Outputs joined as positions 3, 0, 4, 1, 2: routed position p reads joined row inv[p].
    var inv: [5]u32 = undefined;
    invertPositions(&.{ 3, 0, 4, 1, 2 }, &inv);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 4, 0, 2 }, &inv);
}

test "dsv41 experts: a decode-width call never takes the wide lane: construction proves it fits a route" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 64, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    src.trace = &g;
    RecRoute.source = &src;
    defer RecRoute.source = null;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    var rrs = [_]RecRoute{RecRoute.init(a)};
    defer rrs[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    // A top-k whose decode block would overflow one route is refused at construction.
    var wide_k = c;
    wide_k.n_experts_per_tok = 7;
    try testing.expectError(error.DecodeRowsWiderThanRoute, Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &wide_k));
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c);
    defer ex.deinit();
    // 8 rows x top-6 = 48 ids: a decode-width call (the dispatch sends it to the decode lane).
    try testing.expectEqual(@as(u32, 48), decode_forward_rows * 6);
    try testing.expectEqual(@as(usize, 0), rrs[0].calls.items.len);
}

test "dsv41 experts: JOINLESS hands out the wide call's outputs unjoined, each assignment at one (output, row)" {
    const a = testing.allocator;
    var c = testConfig(5120, 2304, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = 64, .rows = &.{64} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, xq.DigXPrefill(TraceOps));
    var digx = [_]xq.DigXPrefill(TraceOps){try xq.DigXPrefill(TraceOps).init(a, &reg, .tier, null)};
    defer digx[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &digx }, &c);
    defer ex.deinit();
    // 40 tokens x top-6 over 60 experts (two groups of at most 48): a deterministic spread.
    const n: u32 = 40;
    const k: u32 = 6;
    var rows: [40 * 6]u16 = undefined;
    for (&rows, 0..) |*e, i| e.* = @intCast((i * 7 + i / 6) % 60);
    var script: Script = .{ .calls = &.{&rows} };
    g.host_values = script.values();
    const parts = try ex.at(0).routedParts(&g, try g.input(&.{ @intCast(n), 5120 }, .bfloat16), try g.input(&.{ @intCast(n), @intCast(k) }, .int32));
    try testing.expect(parts.outs.len >= 2 and parts.outs.len <= Ex.max_parts);
    try testing.expect(g.shapeOf(parts.loc).eql(ops.Shape.of(&.{ @intCast(n), @intCast(k), 2 })));
    // Every (output, row) is some assignment's, once.
    var seen = std.AutoHashMap(u64, void).init(a);
    defer seen.deinit();
    const loc = ex.wide.loc.items;
    for (0..n * k) |q| {
        const s_: usize = @intCast(loc[2 * q]);
        const r_: i32 = loc[2 * q + 1];
        try testing.expect(s_ < parts.outs.len and r_ >= 0 and r_ < g.shapeOf(parts.outs[s_]).dim(0));
        try testing.expect(!(try seen.getOrPut((@as(u64, s_) << 32) | @as(u64, @intCast(r_)))).found_existing);
    }
    var total: c_int = 0;
    for (parts.outs) |o| total += g.shapeOf(o).dim(0);
    try testing.expectEqual(@as(c_int, @intCast(n * k)), total);
    ex.at(0).releaseParts(&g);
}

test "dsv41 experts: a wide call runs the DIG-X prefill route with the lane samples' wave structure" {
    const a = testing.allocator;
    const Sample = struct {
        cases: []const struct {
            case: []const u8,
            calls: []const struct { name: []const u8, a_rows: u32, route: struct { seed: u64, slots: []const u32, counts: []const u32 }, events: []const []const u8 },
        },
    };
    const parsed = try std.json.parseFromSlice(Sample, a, @embedFile("fixtures/dsv41_prefill_wave_samples.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const tier = for (parsed.value.cases) |cs| {
        if (std.mem.eql(u8, cs.case, "tier")) break cs;
    } else return error.NoTierCase;
    var c = testConfig(5120, 2304, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = 64, .rows = &.{64} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, xq.DigXPrefill(TraceOps));
    var digx = [_]xq.DigXPrefill(TraceOps){try xq.DigXPrefill(TraceOps).init(a, &reg, xq.PrefillShape.record3, null)};
    defer digx[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &digx }, &c);
    defer ex.deinit();
    // The sample calls one route serves whole (at most max_route_ids distinct experts),
    // their rows as tokens x k: the lane's slots are this call's experts.
    const Pick = struct { name: []const u8, k: u32 };
    const picks = [_]Pick{ .{ .name = "chunk183", .k = 6 }, .{ .name = "drained", .k = 6 }, .{ .name = "solo_carried", .k = 4 }, .{ .name = "budget_gt", .k = 3 } };
    var checked: usize = 0;
    for (picks) |pk| {
        const cl = for (tier.calls) |x| {
            if (std.mem.eql(u8, x.name, pk.name)) break x;
        } else return error.NoSampleCall;
        try testing.expect(cl.route.slots.len <= max_route_ids);
        const rows = try sampleRows(a, cl.route.seed, cl.route.slots, cl.route.counts);
        defer a.free(rows);
        const n: u32 = @intCast(rows.len / pk.k);
        try testing.expectEqual(rows.len, n * pk.k);
        var script: Script = .{ .calls = &.{rows} };
        g.host_values = script.values();
        const first_node = g.nodes.items.len;
        const resets_before = g.freed.items.len;
        const calls_before = src.counters.route_calls;
        const out = try ex.at(0).routed(&g, try g.input(&.{ @intCast(n), 5120 }, .bfloat16), try g.input(&.{ @intCast(n), @intCast(pk.k) }, .int32));
        try testing.expect(g.shapeOf(out).eql(ops.Shape.of(&.{ @intCast(n), @intCast(pk.k), 5120 })));
        try testing.expectEqual(calls_before + 1, src.counters.route_calls);
        // The lane's waves: 5 launches each (7 outputs: take2 2, gate|up GEMM 2, onepass,
        // down GEMM, widen1), a reset per wave plus the join's.
        var launches: usize = 0;
        for (cl.events) |e| launches += @intFromBool(std.mem.startsWith(u8, e, "launch "));
        const waves = launches / 5;
        var kernels: usize = 0;
        for (g.nodes.items[first_node..]) |nd| kernels += @intFromBool(nd.op == .kernel);
        try testing.expectEqual(7 * waves, kernels);
        try testing.expectEqual(waves + 1, g.freed.items.len - resets_before);
        // A prefill route builds each launch per call (its rows vary up to 2^20).
        try testing.expectEqual(@as(usize, 0), g.prepared_launches);
        checked += 1;
        src.flush() catch {};
    }
    try testing.expectEqual(picks.len, checked);
}

const SynthBank = expert_bank.SynthBank;

/// Every routed id's slot row holds its record's bytes (the rows the hook bound).
fn expectRowsHold(s: *expert_stream.Stream, sb: *const SynthBank, layer: u32, ids: []const u16, sv: Served) !void {
    const geom = &sb.bank.layers[layer];
    const r = for (&s.routes) |*rr| {
        if (rr.state != .free and rr.layer == layer and rr.plan.n_ids == ids.len) break rr;
    } else return error.NoRoute;
    for (ids, r.plan.slotsOf(), sv.refs) |e, slot, ref| {
        try testing.expectEqual(s.slotRef(layer, slot), ref);
        const off = sb.bank.recordOffset(layer, e);
        for (geom.segments, 0..) |seg, comp| {
            try testing.expectEqualSlices(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(layer, slot, @fromBackingInt(@intCast(comp)))[0..seg.length]);
        }
    }
}

test "dsv41 experts: the transient release nulls every layer's binding before the stream frees the scratch, and the grow binds window 0" {
    const a = testing.allocator;
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 5 * 12, .wide_depth = 5, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .transient_release = true });
    defer s.deinit();
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(64, 32, 2);
    var ex = try Experts(TraceOps, StreamSource, TraceMath).init(a, &g, &src, .{ .hidden = 64, .inter = 32 }, &c);
    defer ex.deinit();
    for (ex.banks) |b| try testing.expect(g.shapeOf(b[@backingInt(BankKind.transient)].?.down.rin).eql(ops.Shape.of(&.{ 60, 32 })));
    // The one holder outside the stream (the binding table) is cleared in the call that frees the scratch.
    try testing.expect(try ex.releaseTransient() > 0);
    for (ex.banks, 0..) |b, l| {
        try testing.expect(b[@backingInt(BankKind.transient)] == null);
        try testing.expectEqual(@as(u32, 0), src.bankRows(@intCast(l), .transient));
        try testing.expect(try src.bankArrays(&g, @intCast(l), .transient) == null);
    }
    try testing.expectEqual(@as(u8, 1), src.wideDepth());
    // The grow binds decode's window 0 in every layer (the grown banks' check then sees it).
    try ex.grow(&g, &.{ 4, 8 });
    for (ex.banks) |b| try testing.expect(g.shapeOf(b[@backingInt(BankKind.transient)].?.down.rin).eql(ops.Shape.of(&.{ 12 + expert_stream.decode_staging_rows, 32 })));
}

test "dsv41 experts: the stream adapter hands the hook every routed record's rows, parts after their reads" {
    const a = testing.allocator;
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 } });
    defer s.deinit();
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(64, 32, 2);
    const Ex = Experts(TraceOps, StreamSource, TraceMath);
    var ex = try Ex.init(a, &g, &src, .{ .hidden = 64, .inter = 32 }, &c);
    defer ex.deinit();
    // Base rows and the shared transient bank are bound in the stream's geometry.
    const base = ex.banks[0][@backingInt(BankKind.base)].?;
    try testing.expect(g.shapeOf(base.gate.code).eql(ops.Shape.of(&.{ 4, 4, 2, 48 })));
    try testing.expect(g.shapeOf(ex.banks[1][@backingInt(BankKind.transient)].?.down.rin).eql(ops.Shape.of(&.{ 12, 32 })));

    var script: Script = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 9, 1, 12, 2, 20, 21 }, &.{ 20, 12, 9, 22, 23, 1 } } };
    g.host_values = script.values();
    const xf = try g.input(&.{ 3, 64 }, .bfloat16);
    const idx = try g.input(&.{ 3, 2 }, .int32);
    // Prefill route (one part: the pool's job chunk); every row it serves holds its record.
    _ = try ex.at(0).routed(&g, xf, idx);
    try expectRowsHold(s, &sb, 0, script.calls[0], src.served(&src.calls[0]));
    try testing.expectEqual(@as(u32, 1), src.served(&src.calls[0]).n_parts);
    // Growth binds layer 1's grown rows; decode routes cut parts of <= 3 records.
    try ex.grow(&g, &.{ 4, 8 });
    try testing.expect(g.shapeOf(ex.banks[1][@backingInt(BankKind.ext)].?.up.rout).eql(ops.Shape.of(&.{ 4, 32 })));
    try testing.expect(ex.banks[0][@backingInt(BankKind.ext)] == null);
    _ = try ex.at(1).routed(&g, xf, idx);
    const r1 = src.calls[
        for (src.calls, 0..) |cl, i| {
            if (cl.route != null and cl.route.?.state == .released and cl.route.?.layer == 1) break i;
        } else unreachable
    ];
    try expectRowsHold(s, &sb, 1, script.calls[1], src.served(&r1));
    try testing.expectEqual(@as(u32, 2), src.served(&r1).n_parts); // six misses: 1, 2, 9 | 12, 20, 21
    _ = try ex.at(1).routed(&g, xf, idx);
    const r2 = src.calls[
        for (src.calls, 0..) |cl, i| {
            if (cl.route != null and cl.route.?.state == .released and cl.route.?.layer == 1 and cl.route.?.plan.n_hits > 0) break i;
        } else unreachable
    ];
    const sv2 = src.served(&r2);
    try expectRowsHold(s, &sb, 1, script.calls[2], sv2);
    // 20, 12, 9, 1 are resident (wave 0); 22 and 23 load in one part.
    try testing.expectEqualSlices(u8, &.{ 0, 0, 0, 1, 1, 0 }, sv2.waves);
    try ex.flush();
    const st = src.stats();
    try testing.expectEqual(@as(u64, 3), st.route_calls);
    try testing.expectEqual(@as(u64, 4), st.expert_cache_hits);
}

// DSV41_PHASE1_ROUTE_FIXTURE=<json from the reference runtime's dump_phase1_route_fixture.py>
test "dsv41 experts: the recorded trace's 3,600 decode calls run through the hook as the Python bank serves them" {
    const path = std.mem.span(env.get(.DSV41_PHASE1_ROUTE_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const FixPlan = @import("expert/policy_test.zig").FixPlan;
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        records_per_part: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        seed_plans: []const []const FixPlan,
        routes: []const FixPlan,
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = f.experts, .rows = f.prefill_capacity, .transient_rows = f.transient, .records_per_part = f.records_per_part });
    defer src.deinit();
    // Prefill: the seeds and their waves, as the Python bank admitted them.
    for (0..f.layers) |l| {
        try src.seedPrefill(@intCast(l), f.resident0[l]);
        for (f.seed_plans[l]) |want| {
            const call = try src.route(@intCast(l), want.ids, &.{});
            try @import("expert/policy_test.zig").expectPlan(&call.plan, want);
            src.release(call);
        }
    }
    var g = TraceOps.init(a);
    defer g.deinit();
    var c = testConfig(5120, 2304, @intCast(f.layers));
    c.swiglu_limit = 10.0;
    const Chain = EagerChain(TraceOps, TraceGemv);
    var ex = try Experts(TraceOps, FakeSource, Chain).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, f.decode_capacity);
    const Replay = struct {
        routes: []const FixPlan,
        next: usize = 0,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.routes[self.next].ids);
            self.next += 1;
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
    };
    var replay: Replay = .{ .routes = f.routes };
    g.host_values = .{ .ctx = &replay, .ids = Replay.ids, .argmax = Replay.argmax };
    var xs: [9]u32 = undefined;
    var is: [9]u32 = undefined;
    for (1..9) |rows| {
        xs[rows] = try g.input(&.{ @intCast(rows), 5120 }, .bfloat16);
        is[rows] = try g.input(&.{ @intCast(rows), 6 }, .int32);
    }
    var n_parts: usize = 0;
    var n_groups: usize = 0;
    for (f.routes, 0..) |want, i| {
        const l: u32 = @intCast(i % f.layers);
        const rows = want.ids.len / 6;
        const y = try ex.at(l).routed(&g, xs[rows], is[rows]);
        try testing.expect(g.shapeOf(y).eql(ops.Shape.of(&.{ @intCast(rows), 6, 5120 })));
        const call = for (&src.calls) |*cl| {
            if (cl.state == .released) break cl;
        } else unreachable;
        @import("expert/policy_test.zig").expectPlan(&call.plan, want) catch |e| {
            std.debug.print("route {d} (cycle {d}, layer {d}) differs\n", .{ i, i / f.layers, l });
            return e;
        };
        // The waves are the recorded parts: part p's experts are the ids of wave p + 1.
        const sv = src.served(call);
        try testing.expectEqual(want.parts.len, sv.n_parts);
        for (want.parts, 1..) |part, w| {
            for (want.ids, sv.waves) |e, wv| {
                const in_part = std.mem.indexOfScalar(u16, part, e) != null;
                try testing.expectEqual(in_part, wv == w);
            }
        }
        for (want.ids, sv.waves) |e, wv| try testing.expectEqual(std.mem.indexOfScalar(u16, want.hits, e) != null, wv == 0);
        n_parts += sv.n_parts;
        for (0..sv.n_parts + 1) |w| n_groups += groupsOf(sv, @intCast(w));
    }
    try testing.expectEqual(f.routes.len, replay.next);
    try ex.flush();
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
    std.debug.print("dsv41 experts: {d} recorded decode calls through the hook, {d} parts, {d} bank groups; plans, parts and waves equal the Python bank's\n", .{ f.routes.len, n_parts, n_groups });
}

test "dsv41 experts: VERIFY_ENCODE hoist: the caller's tail is committed behind the hit wave, before any read wait or gate" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2);
    const Chain = EagerChain(TraceOps, TraceGemv);
    // Host waits: call 1 (six misses, two parts), call 2 (three hits, one part), as the decode-call test.
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    var ex = try Experts(TraceOps, FakeSource, Chain).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, &.{ 8, 8 });
    src.log.clearRetainingCapacity();
    src.trace = &g;
    var script: Script = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
    g.host_values = script.values();
    const xf = try g.input(&.{ 2, 256 }, .bfloat16);
    const idx = try g.input(&.{ 2, 3 }, .int32);
    const tail = [_]u32{ try g.input(&.{ 2, 256 }, .float32), try g.input(&.{ 2, 3 }, .float32) };
    const first = g.nodes.items.len;
    _ = try ex.at(0).routedHoist(&g, xf, idx, &tail);
    var kb: [64]u8 = undefined;
    try testing.expectEqualStrings("Rgdgdr", kindsOf(src.log.items, &kb));
    // [<R, R..g0, g0..d0, d0..g1, g1..d1, d1..r, >r]: the tail between the route and the first wait.
    var ae: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .async_eval, &ae);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 1, 1, 1, 1, 0 }, &ae);
    const mark = src.log.items.len;
    const second = g.nodes.items.len;
    _ = try ex.at(0).routedHoist(&g, xf, idx, &tail);
    try testing.expectEqualStrings("fRgdr", kindsOf(src.log.items[mark..], &kb));
    // [<f, f..R, R..g0, g0..d0, d0..r, >r]: the hit wave's commit, then the tail's, then the reads.
    var ae2: [6]usize = undefined;
    opsBetween(&g, src.log.items[mark..], second, .async_eval, &ae2);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 2, 1, 1, 0 }, &ae2);
    var last_kernel: usize = 0;
    var commits: [2]usize = undefined;
    var n_commits: usize = 0;
    const r_at = src.log.items[mark + 1].at;
    const g_at = src.log.items[mark + 2].at;
    for (g.nodes.items[r_at..g_at], r_at..) |nd, i| switch (nd.op) {
        .kernel => last_kernel = i,
        .async_eval => {
            commits[n_commits] = i;
            n_commits += 1;
        },
        else => {},
    };
    // Every hit-wave kernel is built before the first commit (the hit wave's); the tail's is second.
    try testing.expect(last_kernel < commits[0] and commits[0] < commits[1]);
    try ex.flush();

    // Gated: the tail is committed before the gate is registered (and before any wave is waited).
    var gsrc = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer gsrc.deinit();
    var gg = TraceOps.init(a);
    defer gg.deinit();
    var gex = try ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true }).init(a, &gg, &gsrc, Chain.init(.{}, &c), &c);
    defer gex.deinit();
    try gex.grow(&gg, &.{ 8, 8 });
    gsrc.log.clearRetainingCapacity();
    gsrc.trace = &gg;
    var gscript: Script = .{ .calls = &.{&.{ 1, 2, 3, 4, 5, 6 }} };
    gg.host_values = gscript.values();
    const gtail = [_]u32{try gg.input(&.{ 2, 256 }, .float32)};
    const gfirst = gg.nodes.items.len;
    _ = try gex.at(0).routedHoist(&gg, try gg.input(&.{ 2, 256 }, .bfloat16), try gg.input(&.{ 2, 3 }, .int32), &gtail);
    try testing.expectEqualStrings("REr", kindsOf(gsrc.log.items, &kb));
    var gae: [4]usize = undefined;
    opsBetween(&gg, gsrc.log.items, gfirst, .async_eval, &gae);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 0, 0 }, &gae);
    try gex.flush();
}

/// One route's two decode calls (as the VERIFY_ENCODE hoist test's) through a hook built with or without HOIST_FIRST:
/// the trace's nodes other than commits, the source's log, and per call the commits before its first host read and
/// in all.
const HoistRun = struct {
    ops: std.ArrayList(ops.Op) = .empty,
    shapes: std.ArrayList(ops.Shape) = .empty,
    kinds: [32]u8 = undefined,
    n_kinds: usize = 0,
    before_read: [2]usize = .{ 0, 0 },
    commits: [2]usize = .{ 0, 0 },
    bound: bool = false,

    fn deinit(r: *HoistRun, a: std.mem.Allocator) void {
        r.ops.deinit(a);
        r.shapes.deinit(a);
    }

    const Values = struct {
        calls: []const []const u16,
        next: usize = 0,
        fn values(self: *Values) TraceOps.HostValues {
            return .{ .ctx = self, .ids = ids, .argmax = argmax, .f32s = f32s };
        }
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *Values = @ptrCast(@alignCast(ctx));
            if (self.next >= self.calls.len) return error.ScriptExhausted;
            const c = self.calls[self.next];
            self.next += 1;
            if (c.len != out.len) return error.ScriptShape;
            @memcpy(out, c);
        }
        fn f32s(_: *anyopaque, out: []f32) anyerror!void {
            @memset(out, 0);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
    };

    fn run(comptime R: Routes, a: std.mem.Allocator, hoist_first: bool, out: *HoistRun) !void {
        var c = testConfig(256, 128, 2);
        c.n_routed_experts = 16;
        const Chain = EagerChain(TraceOps, TraceGemv);
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
        defer src.deinit();
        if (R.lookahead) src.selector = try expert_lookahead.Selector.init(a, 16, 6, std.math.inf(f32), 2);
        var g = TraceOps.init(a);
        defer g.deinit();
        const Ex = ExpertsWith(TraceOps, FakeSource, Chain, R);
        var gates: [2]Ex.Gate = undefined;
        for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 16, 256 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
        var ex = try Ex.initWith(a, &g, &src, Chain.init(.{}, &c), &c, .{ .gates = if (R.lookahead) &gates else &.{}, .hoist_first = hoist_first });
        defer ex.deinit();
        out.bound = ex.hoistFirst();
        try ex.grow(&g, &.{ 8, 8 });
        src.log.clearRetainingCapacity();
        src.trace = &g;
        // Call 1: six misses (two parts); call 2: three hits, one part.
        var vals: Values = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
        g.host_values = vals.values();
        const xf = try g.input(&.{ 2, 256 }, .bfloat16);
        const idx = try g.input(&.{ 2, 3 }, .int32);
        const tail = [_]u32{ try g.input(&.{ 2, 256 }, .float32), try g.input(&.{ 2, 3 }, .float32) };
        const first = g.nodes.items.len;
        var starts: [3]usize = undefined;
        for (0..2) |call| {
            starts[call] = g.nodes.items.len;
            _ = try ex.at(0).routedHoist(&g, xf, idx, &tail);
        }
        starts[2] = g.nodes.items.len;
        for (0..2) |call| {
            var read_seen = false;
            for (g.nodes.items[starts[call]..starts[call + 1]]) |nd| switch (nd.op) {
                .async_eval => {
                    out.commits[call] += 1;
                    if (!read_seen) out.before_read[call] += 1;
                },
                .host_read => read_seen = true,
                else => {},
            };
        }
        for (g.nodes.items[first..]) |nd| if (nd.op != .async_eval) {
            try out.ops.append(a, nd.op);
            try out.shapes.append(a, nd.shape);
        };
        out.n_kinds = kindsOf(src.log.items, &out.kinds).len;
        try ex.flush();
    }
};

test "dsv41 experts: HOIST_FIRST: the same evals of the same arrays; the hoist committed behind the barrier's arrays, before the wait" {
    const a = testing.allocator;
    inline for (.{ Routes{}, Routes{ .lookahead = true }, Routes{ .gated = true }, Routes{ .gated = true, .lookahead = true } }) |R| {
        var stock: HoistRun = .{};
        defer stock.deinit(a);
        var first: HoistRun = .{};
        defer first.deinit(a);
        try HoistRun.run(R, a, false, &stock);
        try HoistRun.run(R, a, true, &first);
        // Bound at construction; off by default.
        try testing.expect(!stock.bound and first.bound);
        // Exact by construction: every node other than a commit (kernels, glue, host reads) in the same order and shape,
        // and the source sees the same routes, waits and releases.
        try testing.expectEqualSlices(ops.Op, stock.ops.items, first.ops.items);
        for (stock.shapes.items, first.shapes.items) |x, y| try testing.expect(x.eql(y));
        try testing.expectEqualStrings(stock.kinds[0..stock.n_kinds], first.kinds[0..first.n_kinds]);
        for (0..2) |call| {
            // Stock: nothing committed before the barrier's wait (its eval commits the barrier itself).
            try testing.expectEqual(@as(usize, 0), stock.before_read[call]);
            // HOIST_FIRST: the barrier's arrays, then the hoist, both before the wait; no commit of the hoist later.
            try testing.expectEqual(@as(usize, 2), first.before_read[call]);
            try testing.expectEqual(stock.commits[call] + 1, first.commits[call]);
        }
    }
    // A call with no hoist (a caller without one) commits only the barrier's arrays ahead of the wait.
    const c = testConfig(256, 128, 2);
    const Chain = EagerChain(TraceOps, TraceGemv);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    var ex = try Experts(TraceOps, FakeSource, Chain).initWith(a, &g, &src, Chain.init(.{}, &c), &c, .{ .hoist_first = true });
    defer ex.deinit();
    try ex.grow(&g, &.{ 8, 8 });
    var script: Script = .{ .calls = &.{&.{ 1, 2, 3, 4, 5, 6 }} };
    g.host_values = script.values();
    const at = g.nodes.items.len;
    _ = try ex.at(0).routedHoist(&g, try g.input(&.{ 2, 256 }, .bfloat16), try g.input(&.{ 2, 3 }, .int32), &.{});
    var n_before: usize = 0;
    for (g.nodes.items[at..]) |nd| {
        if (nd.op == .host_read) break;
        if (nd.op == .async_eval) n_before += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_before);
    try ex.flush();
}

test "dsv41 experts: LOOKAHEAD4's construction check: the gated waves against the same slots waited, a mismatch or no miss refused" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2);
    const Chain = EagerChain(TraceOps, TraceGemv);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    var ex = try ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true }).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, &.{ 8, 8 });
    src.log.clearRetainingCapacity();
    src.trace = &g;
    // The two outputs each check reads (gated, then waited), scripted.
    const Reads = struct {
        same: [2 * 6 * 256]f32 = @splat(0.25),
        n: usize = 0,
        differ: bool = false,
        fn values(self: *@This()) TraceOps.HostValues {
            return .{ .ctx = self, .ids = ids, .argmax = argmax, .f32s = f32s };
        }
        fn ids(_: *anyopaque, _: []u16) anyerror!void {
            return error.NoIds;
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
        fn f32s(ctx: *anyopaque, out: []f32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.same[0..out.len]);
            if (self.differ and self.n % 2 == 1) out[out.len - 1] = -1;
            self.n += 1;
        }
    };
    var reads: Reads = .{};
    g.host_values = reads.values();
    const xf = try g.input(&.{ 1, 256 }, .bfloat16);
    var gated: [6 * 256]f32 = undefined;
    var waited: [6 * 256]f32 = undefined;
    // Six cold experts: the gate, then each part's waits for the reference waves, release, flush.
    const w0 = g.waits.items.len;
    try ex.checkGated(&g, 0, xf, &.{ 1, 2, 3, 4, 5, 6 }, &gated, &waited);
    var kb: [32]u8 = undefined;
    try testing.expectEqualStrings("REgdgdrf", kindsOf(src.log.items, &kb));
    try testing.expect(g.waits.items.len > w0);
    try testing.expectEqual(@as(usize, 2), reads.n);
    // Six other cold experts whose waited output differs in one float: refused by name.
    reads.differ = true;
    try testing.expectError(error.GateCheckMismatch, ex.checkGated(&g, 0, xf, &.{ 7, 8, 9, 10, 11, 12 }, &gated, &waited));
    // Resident experts only: nothing is gated, nothing checked.
    try testing.expectError(error.GateCheckNoMiss, ex.checkGated(&g, 0, xf, &.{ 7, 8, 9, 10, 11, 12 }, &gated, &waited));
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
}

// DSV41_PHASE0B_MLX=1, on a GPU machine only (any MLX array allocates through Metal, on a CPU stream too).
test "dsv41 experts 0b: a gated wait over handles left as garbage (waitProj's undefined outs) aliases its inputs" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try ops.MlxOps.init(testing.allocator, stream);
    defer g.deinit();
    // A host event already past the gate's value: the CPU consumer runs the WaitEvent without blocking.
    const word = try testing.allocator.create(i64);
    defer testing.allocator.destroy(word);
    word.* = 7;
    const ev = try expert_event.createHost(word, std.time.ns_per_s);
    const vals = [_]f32{ 1, 2, 3, 4 };
    const x = try g.hostArray(std.mem.sliceAsBytes(&vals), &.{4}, .float32);
    const y = try g.hostArray(std.mem.sliceAsBytes(&vals), &.{ 2, 2 }, .float32);
    var outs: [2]mlx.mlx_array = undefined;
    for (&outs) |*o| o.ctx = @ptrFromInt(0xaaaa_aaaa_aaaa_aaa8);
    try mlxEventWait(&g, ev, &.{ x, y }, 5, &.{}, &outs);
    var got: [4]f32 = undefined;
    _ = try g.hostF32(try g.add(outs[0], try g.reshape(outs[1], &.{4})), &got);
    try testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, &got);
    g.reset();
}

test "dsv41 experts: the gated route builds every wave at once over event-wait aliases, no host waits" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    var ex = try ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true }).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, &.{ 8, 8 });
    src.log.clearRetainingCapacity();
    src.trace = &g;
    var script: Script = .{ .calls = &.{&.{ 1, 2, 3, 4, 5, 6 }} };
    g.host_values = script.values();
    const first = g.nodes.items.len;
    _ = try ex.at(0).routed(&g, try g.input(&.{ 2, 256 }, .bfloat16), try g.input(&.{ 2, 3 }, .int32));
    var kb: [16]u8 = undefined;
    try testing.expectEqualStrings("REr", kindsOf(src.log.items, &kb));
    const sv = src.served(&src.calls[0]);
    // Gate/up arrays waited once per miss bank (6 each); each part's down arrays (3 per bank) at its own value.
    var kinds: [n_banks]bool = @splat(false);
    for (sv.waves, sv.refs) |w, r| {
        if (w > 0) kinds[@backingInt(r.bank)] = true;
    }
    var n_kinds: usize = 0;
    for (kinds) |k| n_kinds += @intFromBool(k);
    var n_alias: usize = 0;
    var n_kernel: usize = 0;
    for (g.nodes.items[first..]) |nd| {
        n_alias += @intFromBool(nd.op == .event_wait);
        n_kernel += @intFromBool(nd.op == .kernel);
    }
    try testing.expectEqual(6 * n_kinds + 3 * n_kinds * sv.n_parts, n_alias);
    try testing.expectEqual(3 * (groupsOf(sv, 1) + groupsOf(sv, 2)), n_kernel);
    // One gate per call: its gate/up value and one down value per part.
    try testing.expectEqual(@as(u64, 1 + sv.n_parts), src.gate_value);
    // In build order: every gate/up array at the gate/up value with no deps, then
    // part p's down arrays at down_first + p after all gate/up outputs (+ part p-1's downs).
    const gv: expert_stream.Gates = .{ .gu = 1, .down_first = 2, .n_parts = sv.n_parts }; // the FakeSource's first gate
    const ws = g.waits.items;
    try testing.expectEqual(n_alias, ws.len);
    for (ws[0 .. 6 * n_kinds]) |w| try testing.expectEqual(TraceOps.Wait{ .value = gv.gu, .n_deps = 0 }, w);
    var prev_deps: u32 = 0;
    for (0..sv.n_parts) |p| {
        const part = ws[6 * n_kinds + 3 * n_kinds * p ..][0 .. 3 * n_kinds];
        for (part) |w| {
            try testing.expectEqual(gv.down_first + p, w.value);
            try testing.expectEqual(part[0].n_deps, w.n_deps);
        }
        if (p > 0) try testing.expect(part[0].n_deps > prev_deps);
        prev_deps = part[0].n_deps;
    }
}

/// One routed call of the gated test scheduler (FakeSource, the eager chain), stock or MERGE_GATE_WAITS: its nodes, its
/// waits' aliases, the waits it built, its parts and the banks its misses read.
const MergeWaitsRun = struct {
    nodes: []TraceOps.Node = &.{},
    waits: []TraceOps.Wait = &.{},
    calls: usize = 0,
    parts: u32 = 0,
    banks: usize = 0,

    fn deinit(r: *MergeWaitsRun, a: std.mem.Allocator) void {
        a.free(r.nodes);
        a.free(r.waits);
    }

    fn run(a: std.mem.Allocator, merge: bool, ids: []const u16) !MergeWaitsRun {
        const c = testConfig(256, 128, 2);
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
        defer src.deinit();
        var g = TraceOps.init(a);
        defer g.deinit();
        const Chain = EagerChain(TraceOps, TraceGemv);
        var ex = try ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true }).init(a, &g, &src, Chain.init(.{}, &c), &c);
        defer ex.deinit();
        // Off unless bound; bound once, at construction.
        try testing.expect(!ex.mergeGateWaits());
        if (merge) try ex.bindMergeGateWaits();
        try testing.expectEqual(merge, ex.mergeGateWaits());
        try ex.grow(&g, &.{ 8, 8 });
        src.log.clearRetainingCapacity();
        src.trace = &g;
        var script: Script = .{ .calls = &.{ids} };
        g.host_values = script.values();
        const first = g.nodes.items.len;
        const w0 = g.waits.items.len;
        const c0 = g.wait_calls;
        _ = try ex.at(0).routed(&g, try g.input(&.{ 2, 256 }, .bfloat16), try g.input(&.{ 2, 3 }, .int32));
        const sv = src.served(&src.calls[0]);
        var kinds: [n_banks]bool = @splat(false);
        for (sv.waves, sv.refs) |w, r| {
            if (w > 0) kinds[@backingInt(r.bank)] = true;
        }
        var out: MergeWaitsRun = .{ .calls = g.wait_calls - c0, .parts = sv.n_parts };
        for (kinds) |k| out.banks += @intFromBool(k);
        out.nodes = try a.dupe(TraceOps.Node, g.nodes.items[first..]);
        errdefer a.free(out.nodes);
        out.waits = try a.dupe(TraceOps.Wait, g.waits.items[w0..]);
        return out;
    }
};

test "dsv41 experts: MERGE_GATE_WAITS: one event wait per gate value (1 + parts a call, the stock route 2 per miss bank + one per bank a part), every node, alias, value and dep the stock route's; an ungated scheduler refuses it by name" {
    const a = testing.allocator;
    var waited: usize = 0;
    for ([_][]const u16{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 1, 1, 2, 3, 3, 9 } }) |ids| {
        var stock = try MergeWaitsRun.run(a, false, ids);
        defer stock.deinit(a);
        var merged = try MergeWaitsRun.run(a, true, ids);
        defer merged.deinit(a);
        // The same call: the same parts and miss banks.
        try testing.expectEqual(stock.parts, merged.parts);
        try testing.expectEqual(stock.banks, merged.banks);
        try testing.expect(stock.parts > 0 and stock.banks > 0);
        // Exact by construction: every node (each alias, wave and join) in the same order, op, dtype and shape, and every
        // alias at the same value behind the same deps.
        try testing.expectEqual(stock.nodes.len, merged.nodes.len);
        for (stock.nodes, merged.nodes) |x, y| {
            try testing.expectEqual(x.op, y.op);
            try testing.expectEqual(x.dtype, y.dtype);
            try testing.expect(x.shape.eql(y.shape));
        }
        try testing.expectEqualSlices(TraceOps.Wait, stock.waits, merged.waits);
        // The waits built: the stock route's gate and up of each miss bank and each part's down per bank; one per value.
        try testing.expectEqual(2 * stock.banks + stock.banks * stock.parts, stock.calls);
        try testing.expectEqual(1 + merged.parts, merged.calls);
        waited += stock.calls - merged.calls;
    }
    try testing.expect(waited > 0);
    // An ungated scheduler (its parts wait on the host) refuses the route by name.
    const c = testConfig(256, 128, 2);
    const Chain = EagerChain(TraceOps, TraceGemv);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    var ex = try Experts(TraceOps, FakeSource, Chain).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try testing.expectError(error.MergeGateWaitsNeedsGated, ex.bindMergeGateWaits());
    try testing.expect(!ex.mergeGateWaits());
}

test "dsv41 experts: JOIN_INDEXED: a decode call ending before the join's take: every node the stock call's but the take and its reshape, the order the take's index byte for byte, the rows the take's source" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2);
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true });
    for ([_][]const u16{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 1, 1, 2, 3, 3, 9 }, &.{ 7, 8, 9, 10, 11, 12 } }) |ids| {
        var stock_ops: std.ArrayList(ops.Op) = .empty;
        defer stock_ops.deinit(a);
        var idx_ops: std.ArrayList(ops.Op) = .empty;
        defer idx_ops.deinit(a);
        var stock_order: []u8 = &.{};
        defer a.free(stock_order);
        var idx_order: []u8 = &.{};
        defer a.free(idx_order);
        var stock_rows: ops.Shape = .{};
        var idx_rows: ops.Shape = .{};
        var out_shape: ops.Shape = .{};
        for ([_]bool{ false, true }) |indexed| {
            var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
            defer src.deinit();
            var g = TraceOps.init(a);
            defer g.deinit();
            g.record_host = true;
            var ex = try Ex.init(a, &g, &src, Chain.init(.{}, &c), &c);
            defer ex.deinit();
            try ex.grow(&g, &.{ 8, 8 });
            src.log.clearRetainingCapacity();
            src.trace = &g;
            var script: Script = .{ .calls = &.{ids} };
            g.host_values = script.values();
            const xf = try g.input(&.{ 2, 256 }, .bfloat16);
            const idx = try g.input(&.{ 2, 3 }, .int32);
            const first = g.nodes.items.len;
            if (indexed) {
                const j = try ex.at(0).routedHoistIndexed(&g, xf, idx, &.{});
                for (g.nodes.items[first..]) |nd| try idx_ops.append(a, nd.op);
                idx_order = try a.dupe(u8, g.hostBytesOf(j.order).?);
                idx_rows = g.shapeOf(j.rows);
                try testing.expectEqual(ops.Dtype.uint32, g.dtypeOf(j.order));
            } else {
                const out = try ex.at(0).routedHoist(&g, xf, idx, &.{});
                for (g.nodes.items[first..]) |nd| try stock_ops.append(a, nd.op);
                out_shape = g.shapeOf(out);
                // The take's index: the call's last host array; its source: the node before it (the concat, or the one output).
                var last_host: usize = g.nodes.items.len;
                while (last_host > first) : (last_host -= 1) if (g.nodes.items[last_host - 1].op == .host) break;
                stock_order = try a.dupe(u8, g.hostBytesOf(@intCast(last_host - 1)).?);
                stock_rows = g.nodes.items[last_host - 2].shape;
            }
        }
        // The stock call is the indexed call then the take and its reshape to [n, k, hidden].
        try testing.expectEqual(idx_ops.items.len + 2, stock_ops.items.len);
        try testing.expectEqualSlices(ops.Op, idx_ops.items, stock_ops.items[0..idx_ops.items.len]);
        try testing.expectEqual(ops.Op.take, stock_ops.items[idx_ops.items.len]);
        try testing.expectEqual(ops.Op.reshape, stock_ops.items[idx_ops.items.len + 1]);
        try testing.expect(out_shape.eql(ops.Shape.of(&.{ 2, 3, 256 })));
        // The same order (each routed position's row) and the same rows the take reads.
        try testing.expectEqualSlices(u8, stock_order, idx_order);
        try testing.expectEqual(@as(usize, 2 * 3 * 4), idx_order.len);
        try testing.expect(stock_rows.eql(idx_rows));
    }
}

/// A host-only architecture fixture whose router is already transposed and F32.
const StageBPredictorMath = struct {
    pub const predictor_decode_only = true;
    inner: TraceMath,
    pub fn predictorScores(g: *TraceOps, xf: u32, router: u32, correction: u32) !u32 {
        return g.add(try g.sigmoid(try g.matmul(try g.astype(xf, .float32), router)), correction);
    }
    pub fn gateUp(self: *const StageBPredictorMath, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        return self.inner.gateUp(g, x, ids, gate, up);
    }
    pub fn down(self: *const StageBPredictorMath, g: *TraceOps, h: u32, ids: u32, bank: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, bank);
    }
};

test "mimo sdk2 stageB shared: an architecture predictor uses its borrowed transposed router and the DSV default stays exact" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const xf = try g.input(&.{ 1, 64 }, .bfloat16);
    const correction = try g.input(&.{16}, .float32);
    const router = try g.input(&.{ 64, 16 }, .float32);
    const native_start = g.nodes.items.len;
    const Native = ExpertsWith(TraceOps, FakeSource, StageBPredictorMath, .{ .lookahead = true });
    const native = try Native.nextScores(&g, xf, .{ .w = router, .bias = correction });
    const native_ops = [_]ops.Op{ .astype, .matmul, .sigmoid, .add };
    try testing.expectEqual(native_ops.len, g.nodes.items.len - native_start);
    for (g.nodes.items[native_start..], native_ops) |node, op| try testing.expectEqual(op, node.op);
    try testing.expect(g.shapeOf(native).eql(ops.Shape.of(&.{ 1, 16 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(native));

    const dsv_router = try g.input(&.{ 16, 64 }, .bfloat16);
    const dsv_start = g.nodes.items.len;
    const Default = ExpertsWith(TraceOps, FakeSource, TraceMath, .{ .lookahead = true });
    const dsv = try Default.nextScores(&g, xf, .{ .w = dsv_router, .bias = correction });
    const dsv_ops = [_]ops.Op{ .transpose, .matmul, .astype, .scalar, .logaddexp, .sqrt, .add };
    try testing.expectEqual(dsv_ops.len, g.nodes.items.len - dsv_start);
    for (g.nodes.items[dsv_start..], dsv_ops) |node, op| try testing.expectEqual(op, node.op);
    try testing.expect(g.shapeOf(dsv).eql(ops.Shape.of(&.{ 1, 16 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(dsv));
}

test "mimo sdk2 stageB shared: logical M1 prefill builds no prediction and decode predicts only a following layer" {
    const a = testing.allocator;
    var c = testConfig(64, 32, 2);
    c.n_routed_experts = 16;
    var src = try FakeSource.init(a, .{ .hidden = 64, .inter = 32, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    src.selector = try expert_lookahead.Selector.init(a, 16, 8, std.math.inf(f32), 2);
    var g = TraceOps.init(a);
    defer g.deinit();
    const Ex = ExpertsWith(TraceOps, FakeSource, StageBPredictorMath, .{ .lookahead = true });
    var gates: [2]Ex.Gate = undefined;
    for (&gates) |*gate| gate.* = .{ .w = try g.input(&.{ 64, 16 }, .float32), .bias = try g.input(&.{16}, .float32) };
    var ex = try Ex.initWith(a, &g, &src, .{ .inner = .{ .hidden = 64, .inter = 32 } }, &c, .{ .gates = &gates });
    defer ex.deinit();
    const Reads = struct {
        score_reads: u32 = 0,
        fn ids(_: *anyopaque, out: []u16) anyerror!void {
            const values = [_]u16{ 1, 2, 3, 4, 5, 6 };
            try testing.expectEqual(values.len, out.len);
            @memcpy(out, &values);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
        fn f32s(ctx: *anyopaque, out: []f32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.score_reads += 1;
            for (out, 0..) |*score, e| score.* = 15 - @as(f32, @floatFromInt(e));
        }
    };
    var reads: Reads = .{};
    g.host_values = .{ .ctx = &reads, .ids = Reads.ids, .argmax = Reads.argmax, .f32s = Reads.f32s };
    const xf = try g.input(&.{ 1, 64 }, .bfloat16);
    const indices = try g.input(&.{ 1, 6 }, .int32);
    const prefill_start = g.nodes.items.len;
    _ = try ex.at(0).routed(&g, xf, indices);
    try testing.expectEqual(@as(u32, 0), reads.score_reads);
    try testing.expectEqual(@as(usize, 0), src.picks.items.len);
    for (g.nodes.items[prefill_start..]) |node| try testing.expect(node.op != .sigmoid);
    try ex.grow(&g, &.{ 8, 8 });
    _ = try ex.at(0).routed(&g, xf, indices);
    try testing.expectEqual(@as(u32, 1), reads.score_reads);
    try testing.expectEqual(@as(usize, 1), src.picks.items.len);
    try testing.expectEqual(@as(u32, 1), src.picks.items[0].layer);
    _ = try ex.at(1).routed(&g, xf, indices);
    try testing.expectEqual(@as(u32, 1), reads.score_reads);
    try testing.expectEqual(@as(usize, 1), src.picks.items.len);
    try ex.flush();
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
    // The default DSV route keeps its prefill-tail prediction even on a source with a phase API.
    var stock_src = try FakeSource.init(a, .{ .hidden = 64, .inter = 32, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer stock_src.deinit();
    stock_src.selector = try expert_lookahead.Selector.init(a, 16, 8, std.math.inf(f32), 2);
    const Stock = ExpertsWith(TraceOps, FakeSource, TraceMath, .{ .lookahead = true });
    var stock_gates: [2]Stock.Gate = undefined;
    for (&stock_gates) |*gate| gate.* = .{ .w = try g.input(&.{ 16, 64 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
    var stock = try Stock.initWith(a, &g, &stock_src, .{ .hidden = 64, .inter = 32 }, &c, .{ .gates = &stock_gates });
    defer stock.deinit();
    try testing.expect(!stock_src.decoding());
    const stock_start = g.nodes.items.len;
    _ = try stock.at(0).routed(&g, xf, indices);
    try testing.expectEqual(@as(u32, 2), reads.score_reads);
    var sqrt_count: u32 = 0;
    for (g.nodes.items[stock_start..]) |node| sqrt_count += @intFromBool(node.op == .sqrt);
    try testing.expectEqual(@as(u32, 1), sqrt_count);
    try stock.flush();
}

test "dsv41 experts: the read-ahead of layer l is scored by layer l + 1's router, none after the last" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 3);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = ExpertsWith(TraceOps, FakeSource, Chain, .{ .lookahead = true });
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 16, 256 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
    var ex = try Ex.initWith(a, &g, &src, Chain.init(.{}, &c), &c, .{ .gates = &gates });
    defer ex.deinit();
    try testing.expectEqual(gates[1].w, ex.predictorGate(0).?.w);
    try testing.expectEqual(gates[2].w, ex.predictorGate(1).?.w);
    try testing.expect(ex.predictorGate(2) == null);
}

// DSV41_PHASE2_FIXTURE=<json from the reference runtime's dump_phase2_lookahead_fixture.py> (its scores file beside it)
test "dsv41 experts: the scores the hook passes reproduce the streamer's read-ahead picks on the recorded trace" {
    const path = std.mem.span(env.get(.DSV41_PHASE2_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const Call = struct { ids: []const u16, sel: []const []const u16 };
    const Cfg = struct { k: u32, tau: ?f32, budget: u32 };
    const TieCall = struct { call: u32, config: u32, sel: []const u16 };
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        rows: []const u32,
        configs: []const Cfg,
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
    const scores = try a.alloc(f32, raw.len / 4);
    defer a.free(scores);
    for (scores, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));

    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = f.experts, .rows = f.prefill_capacity, .transient_rows = f.transient });
    defer src.deinit();
    const cf = f.configs[0];
    src.selector = try expert_lookahead.Selector.init(a, f.experts, cf.k, cf.tau orelse std.math.inf(f32), cf.budget);
    // The prefill residency the fixture replays: the seed, admitted in waves of the transient width.
    for (0..f.layers) |l| {
        try src.seedPrefill(@intCast(l), f.resident0[l]);
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        var i: usize = 0;
        while (i < sorted.len) : (i += f.transient) {
            const call = try src.route(@intCast(l), sorted[i..@min(i + f.transient, sorted.len)], &.{});
            src.release(call);
        }
    }
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(5120, 2304, @intCast(f.layers));
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = ExpertsWith(TraceOps, FakeSource, Chain, .{ .lookahead = true });
    const gates = try a.alloc(Ex.Gate, f.layers);
    defer a.free(gates);
    for (gates) |*gt| gt.* = .{ .w = try g.input(&.{ @intCast(f.experts), 5120 }, .bfloat16), .bias = try g.input(&.{@intCast(f.experts)}, .float32) };
    var cc = c;
    cc.n_routed_experts = f.experts;
    var ex = try Ex.initWith(a, &g, &src, Chain.init(.{}, &cc), &cc, .{ .gates = gates });
    defer ex.deinit();
    try ex.grow(&g, f.decode_capacity);
    const Replay = struct {
        calls: []const Call,
        scores: []const f32,
        experts: usize,
        next: usize = 0,
        at: usize = 0,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.calls[self.next].ids);
            self.next += 1;
        }
        fn f32s(ctx: *anyopaque, out: []f32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.scores[self.at..][0..out.len]);
            self.at += out.len;
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
    };
    var replay: Replay = .{ .calls = f.calls, .scores = scores, .experts = f.experts };
    g.host_values = .{ .ctx = &replay, .ids = Replay.ids, .argmax = Replay.argmax, .f32s = Replay.f32s };
    var xs: [9]u32 = undefined;
    var is: [9]u32 = undefined;
    for (1..9) |rows| {
        xs[rows] = try g.input(&.{ @intCast(rows), 5120 }, .bfloat16);
        is[rows] = try g.input(&.{ @intCast(rows), 6 }, .int32);
    }
    var tie_i: usize = 0;
    var n_picks: usize = 0;
    for (f.calls, 0..) |call, ci| {
        const l: u32 = @intCast(ci % f.layers);
        const m = f.rows[ci / f.layers];
        const before = src.picks.items.len;
        _ = try ex.at(l).routed(&g, xs[m], is[m]);
        if (l + 1 < f.layers) {
            try testing.expectEqual(before + 1, src.picks.items.len);
            const pick = src.picks.items[before];
            try testing.expectEqual(l + 1, pick.layer);
            var want = call.sel[0];
            if (tie_i < f.tie_calls.len and f.tie_calls[tie_i].call == ci) {
                if (f.tie_calls[tie_i].config == 0) want = f.tie_calls[tie_i].sel;
                while (tie_i < f.tie_calls.len and f.tie_calls[tie_i].call == ci) tie_i += 1;
            }
            testing.expectEqualSlices(u16, want, pick.experts[0..pick.n]) catch |e| {
                std.debug.print("call {d} (layer {d}): read-ahead picks differ\n", .{ ci, l });
                return e;
            };
            n_picks += 1;
        } else try testing.expectEqual(before, src.picks.items.len);
    }
    try testing.expectEqual(f.scores_rows * f.experts, replay.at);
    try ex.flush();
    std.debug.print("dsv41 experts: {d} layer calls through the hook with lookahead scores; {d} read-ahead picks equal the streamer's\n", .{ f.calls.len, n_picks });
}

/// A wide-lane route over a real Stream (trace backend) that names, per call row, the expert
/// whose record its slot holds (verified byte for byte) and its act row; and records the live
/// routes and the evals so far at the call.
const StreamRec = struct {
    a: std.mem.Allocator,
    recs: std.ArrayList(Rec) = .empty,
    finishes: u32 = 0,

    var stream: ?*expert_stream.Stream = null;
    var bank: ?*const SynthBank = null;
    /// The hook's bound banks of the layer (to name a call's bank kind).
    var base_code: u32 = 0;
    var transient_code: u32 = 0;

    const Rec = struct { experts: []u16, act_row: []u32, base: bool, live: u32, evals: usize, slot: []u32 = &.{} };

    fn init(a: std.mem.Allocator) StreamRec {
        return .{ .a = a };
    }

    fn deinit(self: *StreamRec, _: *TraceOps) void {
        for (self.recs.items) |r| {
            self.a.free(r.experts);
            self.a.free(r.act_row);
            self.a.free(r.slot);
        }
        self.recs.deinit(self.a);
    }

    fn expertIn(s: *expert_stream.Stream, sb: *const SynthBank, slot: u32) !u16 {
        const geom = &sb.bank.layers[0];
        for (0..sb.bank.n_experts) |e| {
            const off = sb.bank.recordOffset(0, @intCast(e));
            const held = for (geom.segments, 0..) |seg, c| {
                if (!std.mem.eql(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(0, slot, @fromBackingInt(@intCast(c)))[0..seg.length])) break false;
            } else true;
            if (held) return @intCast(e);
        }
        return error.SlotHoldsNoRecord;
    }

    pub fn call(self: *StreamRec, g: *TraceOps, act: u32, rows: quant.PrefillRows, b: BankArraysOf(u32)) !u32 {
        const s = stream.?;
        const base = b.gate.code == base_code;
        if (!base and b.gate.code != transient_code) return error.UnknownBank;
        const cap = s.layers[0].policy.capacity;
        const experts = try self.a.alloc(u16, rows.slot.len);
        errdefer self.a.free(experts);
        for (rows.slot, experts) |r, *e| e.* = try expertIn(s, bank.?, if (base) r else cap + r);
        var live: u32 = 0;
        for (&s.routes) |*r| live += @intFromBool(r.state == .live);
        const act_rows = try self.a.dupe(u32, rows.act_row.?);
        errdefer self.a.free(act_rows);
        const slots = try self.a.dupe(u32, rows.slot);
        errdefer self.a.free(slots);
        try self.recs.append(self.a, .{ .experts = experts, .act_row = act_rows, .base = base, .live = live, .evals = g.evals.items.len, .slot = slots });
        return g.input(&.{ @intCast(rows.slot.len), g.shapeOf(act).d[1] }, .float32);
    }

    pub fn finish(self: *StreamRec, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

test "dsv41 experts: the wide feed and read-ahead serve every routed row from its record, hottest first, one drain per group, reads ahead" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    // 60 tokens x top-6, expert j of a token in band j (distinct in a row): 64 experts routed 3..15 times.
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| {
        const t = i / k;
        const j = i % k;
        e.* = @intCast(j * 20 + (t * (j + 3)) % 20);
    }
    var count: [128]u32 = @splat(0);
    for (ids) |e| count[e] += 1;
    var hot: [128]u16 = undefined;
    var n_distinct: usize = 0;
    for (0..128) |e| if (count[e] > 0) {
        hot[n_distinct] = @intCast(e);
        n_distinct += 1;
    };
    try testing.expectEqual(@as(usize, 64), n_distinct);
    std.sort.pdq(u16, hot[0..n_distinct], @as([]const u32, &count), struct {
        fn lt(cn: []const u32, x: u16, y: u16) bool {
            return if (cn[x] != cn[y]) cn[x] > cn[y] else x < y;
        }
    }.lt);
    var first_seen: [128]bool = @splat(false);
    var appear: [128]u16 = undefined;
    var n_app: usize = 0;
    for (ids) |e| if (!first_seen[e]) {
        first_seen[e] = true;
        appear[n_app] = e;
        n_app += 1;
    };

    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    for ([_]Wide{ .{}, .{ .seed = true, .hot_first = true }, .{ .depth = 2 }, .{ .seed = true, .hot_first = true, .depth = 2 }, .{ .seed = true, .depth = 2 }, .{ .hot_first = true } }) |wide| {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = wide.depth, .transient_rows = @as(u32, wide.depth) * max_route_ids });
        defer s.deinit();
        StreamRec.stream = s;
        defer StreamRec.stream = null;
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        var rrs = [_]StreamRec{StreamRec.init(a)};
        defer rrs[0].deinit(&g);
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = wide });
        defer ex.deinit();
        StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        const evals0 = g.evals.items.len;
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
        try ex.flush();
        const recs = rrs[0].recs.items;

        // Exact: every routed row once, computed from its own expert's record with its token.
        var seen: [n * k]bool = @splat(false);
        for (recs) |r| for (r.experts, r.act_row) |e, t| {
            const row = for (0..k) |j| {
                const i = t * k + j;
                if (ids[i] == e and !seen[i]) break i;
            } else return error.RowNotRouted;
            seen[row] = true;
        };
        for (seen) |x| try testing.expect(x);
        // Every distinct record read once (the call starts from empty rows).
        try testing.expectEqual(@as(u64, n_distinct * sb.bank.layers[0].logical_bytes), s.stats().expert_bytes_read);
        try testing.expectEqual(@as(u64, 2), s.stats().route_calls);

        // The groups: the first 48 of the feed order (hottest first) or of first appearance.
        const order: []const u16 = if (wide.hot_first) hot[0..n_distinct] else appear[0..n_app];
        var in_first: [128]bool = @splat(false);
        for (order[0..max_route_ids]) |e| in_first[e] = true;
        var second_started = false;
        for (recs) |r| {
            const grp0 = in_first[r.experts[0]];
            for (r.experts) |e| try testing.expectEqual(grp0, in_first[e]);
            if (!grp0) second_started = true else try testing.expect(!second_started);
            // Group 0's waves run with group 1 already routed (its reads issued) at depth 2.
            if (grp0) try testing.expectEqual(@as(u32, wide.depth), r.live) else try testing.expectEqual(@as(u32, 1), r.live);
        }
        // Drains: one per group with the feed, else one per bank call.
        const drains = g.evals.items.len - evals0;
        try testing.expectEqual(if (wide.seed) @as(usize, 2) else recs.len, drains);
        try testing.expectEqual(@as(u32, @intCast(if (wide.seed) 2 else recs.len)), rrs[0].finishes);
        // The feed's seed: the persistent rows hold the 16 hottest (protected), the rest transient.
        if (wide.seed and wide.hot_first) {
            var in_top: [128]bool = @splat(false);
            for (hot[0..16]) |e| in_top[e] = true;
            for (recs) |r| for (r.experts) |e| try testing.expectEqual(in_top[e], r.base);
        }
    }
}

test "dsv41 experts: the route-plan cut (Wide.route_cut) hands the math the control's calls: every bank pass's slots, records, act rows and positions in the same order, the same reads and drains" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    // 120 tokens x top-6 over 3 groups of 48: a hot band (experts 0..9 in every 4th token), the rest spread, the
    // first appearance order unlike the hot order.
    const n = 120;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| {
        const t = i / k;
        const j = i % k;
        e.* = if (t % 4 == 0 and j < 2) @intCast((t / 4 + 5 * j) % 10) else @intCast(10 + j * 19 + (t * (j + 7)) % 19);
    }
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    const Run = struct {
        recs: std.ArrayList(StreamRec.Rec) = .empty,
        pos: std.ArrayList(u32) = .empty,
        bytes: u64 = 0,
        drains: usize = 0,
        finishes: u32 = 0,

        fn deinit(r: *@This(), al: std.mem.Allocator) void {
            for (r.recs.items) |x| {
                al.free(x.experts);
                al.free(x.act_row);
                al.free(x.slot);
            }
            r.recs.deinit(al);
            r.pos.deinit(al);
        }
    };
    for ([_]Wide{ .{}, .{ .seed = true, .hot_first = true, .depth = 2 }, .{ .depth = 2 }, .{ .hot_first = true }, .{ .seed = true, .hot_first = true } }) |base_wide| {
        var runs: [2]Run = .{ .{}, .{} };
        defer for (&runs) |*r| r.deinit(a);
        for ([_]bool{ false, true }, &runs) |cut, *run| {
            var wide = base_wide;
            wide.route_cut = cut;
            const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = wide.depth, .transient_rows = @as(u32, wide.depth) * max_route_ids });
            defer s.deinit();
            StreamRec.stream = s;
            defer StreamRec.stream = null;
            var src = StreamSource.init(s);
            var g = TraceOps.init(a);
            defer g.deinit();
            var rrs = [_]StreamRec{StreamRec.init(a)};
            defer rrs[0].deinit(&g);
            var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = wide });
            defer ex.deinit();
            StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
            StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
            var script: Script = .{ .calls = &.{&ids} };
            g.host_values = script.values();
            const evals0 = g.evals.items.len;
            _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
            try ex.flush();
            run.drains = g.evals.items.len - evals0;
            run.finishes = rrs[0].finishes;
            run.bytes = s.stats().expert_bytes_read;
            try run.pos.appendSlice(a, ex.wide.pos.items);
            // The recorder's calls moved over (its deinit then frees nothing).
            try run.recs.appendSlice(a, rrs[0].recs.items);
            rrs[0].recs.clearRetainingCapacity();
        }
        const want = runs[0];
        const got = runs[1];
        try testing.expect(want.recs.items.len >= 2);
        try testing.expectEqual(want.recs.items.len, got.recs.items.len);
        for (want.recs.items, got.recs.items) |x, y| {
            try testing.expectEqualSlices(u32, x.slot, y.slot);
            try testing.expectEqualSlices(u16, x.experts, y.experts);
            try testing.expectEqualSlices(u32, x.act_row, y.act_row);
            try testing.expectEqual(x.base, y.base);
            try testing.expectEqual(x.live, y.live);
            try testing.expectEqual(x.evals, y.evals);
        }
        try testing.expectEqualSlices(u32, want.pos.items, got.pos.items);
        try testing.expectEqual(want.bytes, got.bytes);
        try testing.expectEqual(want.drains, got.drains);
        try testing.expectEqual(want.finishes, got.finishes);
    }
}

test "dsv41 experts: bucketRowsByGroup keeps every group's rows in row order, with or without the counts" {
    // 3 groups over 5 experts; rows of 2 ids.
    const ids = [_]u16{ 4, 0, 2, 4, 1, 3, 0, 2, 4, 1 };
    const distinct = [_]u16{ 4, 0, 2, 1, 3 };
    const grp_of = [_]u16{ 0, 1, 1, 2, 0 };
    var count: [5]u32 = @splat(0);
    for (ids) |e| count[e] += 1;
    for ([_]?[]const u32{ &count, null }) |cn| {
        var off: [4]u32 = undefined;
        var out: [ids.len]u32 = undefined;
        bucketRowsByGroup(&ids, &distinct, &grp_of, cn, &off, &out);
        try testing.expectEqualSlices(u32, &.{ 0, 5, 9, 10 }, &off);
        // Group 0 = experts 4, 0; group 1 = 2, 1; group 2 = 3: each group's id positions ascending.
        try testing.expectEqualSlices(u32, &.{ 0, 1, 3, 6, 8, 2, 4, 7, 9, 5 }, &out);
    }
}

test "dsv41 experts: P1 at wide depth 3: three groups live, the read-ahead landed before the seed, every routed row served from its record" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    // 60 tokens x top-6, token t's j-th expert (j * 20 + t) mod 120: 120 distinct experts, three groups of the feed.
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i % k) * 20 + i / k) % 120);
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = 3, .transient_rows = 3 * max_route_ids });
    defer s.deinit();
    StreamRec.stream = s;
    defer StreamRec.stream = null;
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    var rrs = [_]StreamRec{StreamRec.init(a)};
    defer rrs[0].deinit(&g);
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true, .hot_first = true, .depth = 3, .read_ahead = true } });
    defer ex.deinit();
    StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
    StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
    // The predicted seed (the model's predictor pass would hand it over): 16 experts, every one routed by the call.
    var predicted: [16]u16 = undefined;
    for (&predicted, 0..) |*e, i| e.* = @intCast(i * 7);
    try ex.at(0).readAheadSeed(&predicted);
    try testing.expect(s.ahead.live and s.ahead.n == 16);
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
    try ex.flush();
    try testing.expect(!s.ahead.live);
    // Exact: every routed row once, computed from its own expert's record with its token.
    var seen: [n * k]bool = @splat(false);
    for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
        const row = for (0..k) |j| {
            const i = t * k + j;
            if (ids[i] == e and !seen[i]) break i;
        } else return error.RowNotRouted;
        seen[row] = true;
    };
    for (seen) |x| try testing.expect(x);
    // Three groups routed, the first group's waves run with all three live (the third window's reads in flight).
    try testing.expectEqual(@as(u64, 3), s.stats().route_calls);
    var max_live: u32 = 0;
    for (rrs[0].recs.items) |r| max_live = @max(max_live, r.live);
    try testing.expectEqual(@as(u32, 3), max_live);
    // The read-ahead's records at the barrier: 16 posted, every one routed by the call; its bytes are the reads'.
    const st = s.stats();
    try testing.expectEqual(@as(u64, 16), st.ahead_posted);
    try testing.expectEqual(@as(u64, 16), st.ahead_hits);
    const rec = sb.bank.layers[0].logical_bytes;
    try testing.expectEqual(16 * rec, st.ahead_bytes);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped + st.ahead_posted) * rec, st.expert_bytes_read);
}

test "dsv41 experts: P1 on seeded rows: a load's seed fills every prompt row, the prompt's read-ahead still posts (replacing the residents it leaves out), every routed row served from its record" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i % k) * 20 + i / k) % 120);
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = 3, .transient_rows = 3 * max_route_ids });
    defer s.deinit();
    StreamRec.stream = s;
    defer StreamRec.stream = null;
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    var rrs = [_]StreamRec{StreamRec.init(a)};
    defer rrs[0].deinit(&g);
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true, .hot_first = true, .depth = 3, .read_ahead = true } });
    defer ex.deinit();
    StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
    StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
    // The load's seed (a usage profile's order): experts 120..127 and 0..7, every row of both layers filled, unprotected.
    const Order = struct {
        fn of(_: void, _: u32) []const u16 {
            return &.{ 120, 121, 122, 123, 124, 125, 126, 127, 0, 1, 2, 3, 4, 5, 6, 7 };
        }
    };
    try testing.expectEqual(@as(u64, 32), try s.seedResidents({}, Order.of));
    try testing.expectEqual(@as(u32, 16), s.layers[0].policy.occupancy);
    // The seed reads through the read-ahead's load path: its 32 records count as posted; the prompt's own are the delta.
    const st0 = s.stats();
    try testing.expectEqual(@as(u64, 32), st0.ahead_posted);
    // The prompt's predicted seed: 0, 7, 14, ..., 105 (0 and 7 already resident: hits to be, never victims; the other
    // fourteen replace the seeded residents 120..127 and 1..6, the coldest first, none routed by this call).
    var predicted: [16]u16 = undefined;
    for (&predicted, 0..) |*e, i| e.* = @intCast(i * 7);
    try ex.at(0).readAheadSeed(&predicted);
    try testing.expect(s.ahead.live);
    try testing.expectEqual(@as(u32, 14), s.ahead.n);
    for (predicted) |e| try testing.expect(s.layers[0].policy.slotOf(e) != null);
    for ([_]u16{ 120, 127, 1, 6 }) |e| try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(e));
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
    try ex.flush();
    try testing.expect(!s.ahead.live);
    // Exact: every routed row once, computed from its own expert's record with its token.
    var seen: [n * k]bool = @splat(false);
    for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
        const row = for (0..k) |j| {
            const i = t * k + j;
            if (ids[i] == e and !seen[i]) break i;
        } else return error.RowNotRouted;
        seen[row] = true;
    };
    for (seen) |x| try testing.expect(x);
    // The read-ahead's records at the barrier: 14 posted and 14 hits (the barrier tallies the admissions; the two
    // predicted experts already resident were hits of the seed's own, not the read-ahead's); its bytes are the reads'.
    const st = s.stats();
    try testing.expectEqual(@as(u64, 14), st.ahead_posted - st0.ahead_posted);
    try testing.expectEqual(@as(u64, 14), st.ahead_hits - st0.ahead_hits);
    const rec = sb.bank.layers[0].logical_bytes;
    try testing.expectEqual(14 * rec, st.ahead_bytes - st0.ahead_bytes);
    const demand = (st.expert_cache_misses - st0.expert_cache_misses) - (st.loads_skipped - st0.loads_skipped);
    try testing.expectEqual((demand + 14) * rec, st.expert_bytes_read - st0.expert_bytes_read);
    // Layer 1's seeded residents are untouched (no prompt read-ahead or route there).
    try testing.expectEqual(@as(u32, 16), s.layers[1].policy.occupancy);
    try testing.expect(s.layers[1].policy.slotOf(120) != null and s.layers[1].policy.slotOf(7) != null);
}

test "dsv41 experts: P1b: the seed's base call drains once the seed has landed, before the next group's read wait; every routed row keeps its slot" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 128;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    // 40 tokens x top-6 over 128 experts, hottest first: the 32 persistent rows take the seed (the first group's
    // top 32); the rest of that group and the two after it stream through the transient window (depth 1).
    const n = 40;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast((7 * (i / k) + 11 * (i % k)) % 128);
    const Row = struct { code: u32, slot: u32, act: u32 };
    var rows: [2]std.ArrayList(Row) = .{ .empty, .empty };
    defer for (&rows) |*r| r.deinit(a);
    for ([_]bool{ false, true }, 0..) |at_seed, run| {
        var reg = try hostRegistry();
        defer reg.deinit();
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 128, .rows = &.{32} });
        defer src.deinit();
        var g = TraceOps.init(a);
        defer g.deinit();
        src.trace = &g;
        RecRoute.source = &src;
        defer RecRoute.source = null;
        var rrs = [_]RecRoute{RecRoute.init(a)};
        defer rrs[0].deinit(&g);
        const wide: Wide = .{ .seed = true, .hot_first = true, .defer_base = true, .base_at_seed = at_seed };
        var ex = try Ex.initWith(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c, .{ .wide = wide });
        defer ex.deinit();
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 256 }, .float32), try g.input(&.{ n, k }, .int32));
        var seen: [n * k]bool = @splat(false);
        for (ex.wide.pos.items) |p| {
            try testing.expect(!seen[p]);
            seen[p] = true;
        }
        try testing.expectEqual(@as(usize, n * k), ex.wide.pos.items.len);
        const base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        var buf: [4096]u8 = undefined;
        const kinds = kindsOf(src.log.items, &buf);
        var n_base: usize = 0;
        for (rrs[0].calls.items, 0..) |rec, ci| {
            for (rec.slot, rec.act_row) |sl, t| try rows[run].append(a, .{ .code = rec.bank_code, .slot = sl, .act = t });
            if (rec.bank_code != base_code) continue;
            n_base += 1;
            if (!at_seed) {
                // Without P1b: after every group's call.
                try testing.expectEqual(rrs[0].calls.items.len - 1, ci);
                continue;
            }
            // P1b: after the seed's group is released and the next route has posted its reads, before that group's
            // read wait and calls.
            try testing.expect(ci + 1 < rrs[0].calls.items.len);
            try testing.expectEqualStrings("rfR", kinds[rec.at - 3 .. rec.at]);
            try testing.expect(std.mem.indexOfScalar(u8, kinds[rec.at..], 'g') != null);
        }
        // One base call either way: the seed's rows (every base row is a seed row here).
        try testing.expectEqual(@as(usize, 1), n_base);
        try testing.expectEqual(@as(usize, 0), src.held.items.len);
    }
    // Every routed row resolves to the same bank and slot either way: the schedule moved, not the routes.
    const lt = struct {
        fn f(_: void, x: Row, y: Row) bool {
            if (x.code != y.code) return x.code < y.code;
            if (x.slot != y.slot) return x.slot < y.slot;
            return x.act < y.act;
        }
    }.f;
    for (&rows) |*r| std.sort.pdq(Row, r.items, {}, lt);
    try testing.expectEqual(rows[0].items.len, rows[1].items.len);
    for (rows[0].items, rows[1].items) |x, y| try testing.expectEqual(x, y);
    // P1b needs the seed and the deferred base call: refused at construction without them.
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 128, .rows = &.{32} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    var rrs = [_]RecRoute{RecRoute.init(a)};
    defer rrs[0].deinit(&g);
    try testing.expectError(error.InvalidWideRoute, Ex.initWith(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c, .{ .wide = .{ .seed = true, .base_at_seed = true } }));
}

test "dsv41 experts: P1b at wide depth 3: the seed's base call runs with the later groups' reads live, every routed row from its record" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    // 60 tokens x top-6, token t's j-th expert (j * 20 + t) mod 120: every expert 3 rows, hottest first = by id; the
    // 16 persistent rows take the seed (experts 0-15), the first group's other 32 and two more groups stream.
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i % k) * 20 + i / k) % 120);
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = 3, .transient_rows = 3 * max_route_ids });
    defer s.deinit();
    StreamRec.stream = s;
    defer StreamRec.stream = null;
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    var rrs = [_]StreamRec{StreamRec.init(a)};
    defer rrs[0].deinit(&g);
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true, .hot_first = true, .depth = 3, .defer_base = true, .base_at_seed = true } });
    defer ex.deinit();
    StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
    StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
    try ex.flush();
    // Exact: every routed row once, computed from its own expert's record with its token.
    var seen: [n * k]bool = @splat(false);
    for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
        const row = for (0..k) |j| {
            const i = t * k + j;
            if (ids[i] == e and !seen[i]) break i;
        } else return error.RowNotRouted;
        seen[row] = true;
    };
    for (seen) |x| try testing.expect(x);
    // The seed's one base call follows the first group's transient call, the two later groups live (reads posted).
    var n_base: usize = 0;
    for (rrs[0].recs.items, 0..) |r, i| if (r.base) {
        n_base += 1;
        try testing.expectEqual(@as(u32, 2), r.live);
        try testing.expect(i >= 1 and !rrs[0].recs.items[i - 1].base and i + 1 < rrs[0].recs.items.len);
    };
    try testing.expectEqual(@as(usize, 1), n_base);
    try testing.expectEqual(@as(u64, 3), s.stats().route_calls);
}

test "dsv41 experts: wide depth 5 (the served default): five groups live, the read-ahead landed before the seed, every routed row served from its record" {
    const a = testing.allocator;
    var sb = try SynthBank.open(256);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 256;
    // 120 tokens x top-6, token t's j-th expert (j * 40 + t) mod 240: every expert 3 rows, hottest first = by id; five
    // groups of the feed, every one routed at the barrier (five windows).
    const n = 120;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i % k) * 40 + i / k) % 240);
    const depth = expert_stream.max_wide_depth;
    try testing.expectEqual(@as(u8, 5), depth);
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 1024 }, .wide_depth = depth, .transient_rows = depth * max_route_ids });
    defer s.deinit();
    StreamRec.stream = s;
    defer StreamRec.stream = null;
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    var rrs = [_]StreamRec{StreamRec.init(a)};
    defer rrs[0].deinit(&g);
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true, .hot_first = true, .depth = depth, .read_ahead = true, .defer_base = true, .base_at_seed = true } });
    defer ex.deinit();
    StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
    StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
    var predicted: [16]u16 = undefined;
    for (&predicted, 0..) |*e, i| e.* = @intCast(i * 7);
    try ex.at(0).readAheadSeed(&predicted);
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
    try ex.flush();
    try testing.expect(!s.ahead.live);
    // Exact: every routed row once, computed from its own expert's record with its token.
    var seen: [n * k]bool = @splat(false);
    for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
        const row = for (0..k) |j| {
            const i = t * k + j;
            if (ids[i] == e and !seen[i]) break i;
        } else return error.RowNotRouted;
        seen[row] = true;
    };
    for (seen) |x| try testing.expect(x);
    // Five groups routed, all live before the first group's waves (the windows' reads in flight).
    try testing.expectEqual(@as(u64, 5), s.stats().route_calls);
    var max_live: u32 = 0;
    for (rrs[0].recs.items) |r| max_live = @max(max_live, r.live);
    try testing.expectEqual(@as(u32, 5), max_live);
}

test "dsv41 experts: P1c: the seed-aligned groups put the seed's base call ahead of every stream wave, every routed row from its record" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    // 60 tokens x top-6 over 120 experts, 3 rows each: hottest first = by id. The 16 persistent rows take the seed
    // (experts 0-15). Aligned: the seed's group (0-15), then the stream's (16-63, 64-111, 112-119). P1b's one run: the
    // first group (0-47) holds the seed and 32 stream records, whose waves precede its base call.
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i % k) * 20 + i / k) % 120);
    for ([_]bool{ false, true }) |aligned| {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = 3, .transient_rows = 3 * max_route_ids });
        defer s.deinit();
        StreamRec.stream = s;
        defer StreamRec.stream = null;
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        var rrs = [_]StreamRec{StreamRec.init(a)};
        defer rrs[0].deinit(&g);
        const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true, .hot_first = true, .depth = 3, .defer_base = true, .base_at_seed = true, .seed_aligned = aligned } });
        defer ex.deinit();
        StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
        try ex.flush();
        try testing.expectEqual(@as(u32, 16), s.seedRanks(0));
        // Exact: every routed row once, computed from its own expert's record with its token.
        var seen: [n * k]bool = @splat(false);
        for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
            const row = for (0..k) |j| {
                const i = t * k + j;
                if (ids[i] == e and !seen[i]) break i;
            } else return error.RowNotRouted;
            seen[row] = true;
        };
        for (seen) |x| try testing.expect(x);
        // The one base call: aligned, the first call of all, with the three stream groups' routes live (their reads
        // posted); P1b's one run, after the first group's stream waves, with the two later groups live.
        const recs = rrs[0].recs.items;
        var bi: usize = recs.len;
        for (recs, 0..) |r, i| if (r.base) {
            try testing.expectEqual(recs.len, bi);
            bi = i;
        };
        try testing.expectEqual(@as(usize, if (aligned) 0 else 1), bi);
        try testing.expectEqual(@as(u32, if (aligned) 3 else 2), recs[bi].live);
        try testing.expectEqual(@as(u64, if (aligned) 4 else 3), s.stats().route_calls);
    }
}

test "dsv41 experts: a read-ahead deeper than the source's windows is refused at construction" {
    const a = testing.allocator;
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 4, 4 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 } });
    defer s.deinit();
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(64, 32, 2);
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    var rrs = [_]StreamRec{ StreamRec.init(a), StreamRec.init(a) };
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    try testing.expectError(error.WideDepthExceedsSource, Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .depth = 2 } }));
    try testing.expectError(error.InvalidWideRoute, Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .depth = expert_stream.max_wide_depth + 1 } }));
    const Plain = ExpertsWith(TraceOps, StreamSource, Math, .{});
    try testing.expectError(error.InvalidWideRoute, Plain.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true } }));
    try testing.expectError(error.InvalidWideRoute, Plain.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .hot_first = true } }));
}

/// Decode-lane math that records each gateUp call's bank, the expert whose record each slot
/// holds at the call (StreamRec's stream and bank) and the act rows (the act-row index is the
/// host array made right before the call's `take`), in calls of <= 8 rows.
const ColdRec = struct {
    inner: TraceMath,
    pub const max_decode_rows: u32 = 8;
    var log: std.ArrayList(Call) = .empty;
    const Call = struct { code: u32, experts: []u16, act_row: []u32, evals: usize };

    fn reset(a: std.mem.Allocator) void {
        for (log.items) |c| {
            a.free(c.experts);
            a.free(c.act_row);
        }
        log.clearAndFree(a);
    }

    pub fn gateUp(self: *const ColdRec, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        const slots = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(g.hostBytesOf(ids) orelse return error.NoHostBytes)));
        const acts = g.hostBytesOf(x - 1) orelse return error.NoHostBytes;
        const s = StreamRec.stream.?;
        const base = gate.code == StreamRec.base_code;
        const experts = try g.gpa.alloc(u16, slots.len);
        errdefer g.gpa.free(experts);
        for (slots, experts) |slot, *e| e.* = try StreamRec.expertIn(s, StreamRec.bank.?, if (base) slot else s.layers[0].policy.capacity + slot);
        try log.append(g.gpa, .{
            .code = gate.code,
            .experts = experts,
            .act_row = try g.gpa.dupe(u32, std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(acts)))),
            .evals = g.evals.items.len,
        });
        return self.inner.gateUp(g, x, ids, gate, up);
    }

    pub fn down(self: *const ColdRec, g: *TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, d);
    }
};

test "dsv41 experts: cold rows run the decode lane over their own records after their bank's wide waves, every row once" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast((i % k) * 20 + ((i / k) * ((i % k) + 3)) % 20);
    var count: [128]u32 = @splat(0);
    for (ids) |e| count[e] += 1;
    const Math = WithPrefillRoutes(TraceOps, ColdRec, StreamRec);
    try testing.expectEqual(@as(u32, 8), Math.max_decode_rows);
    for ([_]Wide{ .{ .cold_rows = 3 }, .{ .cold_rows = 3, .seed = true, .hot_first = true, .depth = 2 } }) |wide| {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = wide.depth, .transient_rows = @as(u32, wide.depth) * max_route_ids });
        defer s.deinit();
        StreamRec.stream = s;
        defer StreamRec.stream = null;
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        g.record_host = true;
        defer ColdRec.reset(a);
        var rrs = [_]StreamRec{StreamRec.init(a)};
        defer rrs[0].deinit(&g);
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .inner = .{ .hidden = 64, .inter = 32 } }, .routes = &rrs }, &c, .{ .wide = wide });
        defer ex.deinit();
        const base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        StreamRec.base_code = base_code;
        StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
        try ex.flush();

        // Every routed row once: the wide route's rows are the hot experts', the decode lane's the cold ones'.
        var seen: [n * k]bool = @splat(false);
        const mark = struct {
            fn f(sn: *[n * k]bool, idv: []const u16, e: u16, t: u32) !void {
                for (0..k) |j| {
                    const i = t * k + j;
                    if (idv[i] == e and !sn[i]) {
                        sn[i] = true;
                        return;
                    }
                }
                return error.RowNotRouted;
            }
        }.f;
        for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
            try testing.expect(count[e] > 3);
            try mark(&seen, &ids, e, t);
        };
        var cold_calls: usize = 0;
        for (ColdRec.log.items) |cl| {
            try testing.expect(cl.experts.len >= 1 and cl.experts.len <= 8);
            for (cl.experts, cl.act_row) |e, t| {
                try testing.expectEqual(@as(u32, 3), count[e]);
                try mark(&seen, &ids, e, t);
            }
            cold_calls += 1;
        }
        for (seen) |x| try testing.expect(x);
        // 40 experts routed 3 times: 120 cold rows in calls of at most 8.
        var cold_rows: usize = 0;
        for (ColdRec.log.items) |cl| cold_rows += cl.experts.len;
        try testing.expectEqual(@as(usize, 120), cold_rows);
        try testing.expect(cold_calls >= 15);
        try testing.expectEqual(@as(u64, 64 * sb.bank.layers[0].logical_bytes), s.stats().expert_bytes_read);
    }
    // A cold threshold past the bound, or without the wide lane, is refused at construction.
    {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 } });
        defer s.deinit();
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        var rrs = [_]StreamRec{StreamRec.init(a)};
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        try testing.expectError(error.InvalidWideRoute, Ex.initWith(a, &g, &src, .{ .d = .{ .inner = .{ .hidden = 64, .inter = 32 } }, .routes = &rrs }, &c, .{ .wide = .{ .cold_rows = Wide.max_cold_rows + 1 } }));
        const Plain = ExpertsWith(TraceOps, StreamSource, Math, .{});
        try testing.expectError(error.InvalidWideRoute, Plain.initWith(a, &g, &src, .{ .d = .{ .inner = .{ .hidden = 64, .inter = 32 } }, .routes = &rrs }, &c, .{ .wide = .{ .cold_rows = 2 } }));
    }
}

/// A wide route whose outputs carry, per row, its act row, slot and bank (host arrays the trace
/// records): `callParts` hands them out in waves of 3 rows, last rows first.
const EncRoute = struct {
    const hidden = 64;

    fn rowsOut(g: *TraceOps, a: std.mem.Allocator, rows: quant.PrefillRows, bank: BankArraysOf(u32), which: []const u32) !u32 {
        const v = try a.alloc(f32, which.len * hidden);
        defer a.free(v);
        @memset(v, 0);
        for (which, 0..) |i, r| {
            v[r * hidden] = @floatFromInt(rows.act_row.?[i]);
            v[r * hidden + 1] = @floatFromInt(rows.slot[i]);
            v[r * hidden + 2] = @floatFromInt(bank.gate.code);
        }
        return g.hostArray(std.mem.sliceAsBytes(v), &.{ @intCast(which.len), hidden }, .float32);
    }

    pub fn call(_: *EncRoute, g: *TraceOps, _: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const idx = try g.gpa.alloc(u32, rows.slot.len);
        defer g.gpa.free(idx);
        for (idx, 0..) |*x, i| x.* = @intCast(i);
        return rowsOut(g, g.gpa, rows, bank, idx);
    }

    pub fn callParts(_: *EncRoute, g: *TraceOps, _: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32), alloc: std.mem.Allocator, outs: *std.ArrayList(u32), pos: *std.ArrayList(u32)) !void {
        var end: usize = rows.slot.len;
        while (end > 0) {
            const start = end -| 3;
            var which: [3]u32 = undefined;
            for (start..end, 0..) |i, j| which[j] = @intCast(i);
            try outs.append(alloc, try rowsOut(g, alloc, rows, bank, which[0 .. end - start]));
            try pos.appendSlice(alloc, which[0 .. end - start]);
            end = start;
        }
    }

    pub fn finish(_: *EncRoute, _: *TraceOps) !void {}
};

test "dsv41 experts: the unjoined prefill hands the combine each assignment's own row: its token, its slot, its bank" {
    const a = testing.allocator;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 30;
    var src = try FakeSource.init(a, .{ .hidden = 64, .inter = 32, .n_experts = 30, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    g.record_host = true;
    const Math = WithPrefillRoutes(TraceOps, TraceMath, EncRoute);
    try testing.expect(Math.has_parts);
    var rr = [_]EncRoute{.{}};
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rr }, &c);
    defer ex.deinit();
    // 10 tokens x top-6 over 30 experts: one group, both banks (16 persistent rows, the rest transient).
    const n: u32 = 10;
    const k: u32 = 6;
    var ids: [10 * 6]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i / k) * 5 + (i % k) * 7) % 30);
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    const parts = try ex.at(0).routedParts(&g, try g.input(&.{ @intCast(n), 64 }, .bfloat16), try g.input(&.{ @intCast(n), @intCast(k) }, .int32));
    defer ex.at(0).releaseParts(&g);
    try testing.expect(parts.outs.len > 2 and parts.outs.len <= Ex.max_parts);
    const call = for (&src.calls) |*cl| {
        if (cl.plan.n_ids > 0 and cl.plan.n_ids == ex.wide.distinct.items.len) break cl;
    } else return error.NoCall;
    const loc = ex.wide.loc.items;
    for (ids, 0..) |e, q| {
        const s_: usize = @intCast(loc[2 * q]);
        const r_: usize = @intCast(loc[2 * q + 1]);
        const bytes = g.hostBytesOf(parts.outs[s_]) orelse return error.NoHostBytes;
        const row = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(bytes)))[r_ * EncRoute.hidden ..][0..3];
        const ref = call.refs[@intCast(ex.wide.first.items[e])];
        try testing.expectEqual(@as(f32, @floatFromInt(q / k)), row[0]);
        try testing.expectEqual(@as(f32, @floatFromInt(ref.row)), row[1]);
        try testing.expectEqual(@as(f32, @floatFromInt(ex.banks[0][@backingInt(ref.bank)].?.gate.code)), row[2]);
    }
}

/// The adjacent runs JOINLESS merged in before the minimal copy (`per = ceil(n / 24)` adjacent outputs per
/// source), as a plan: the reference the merge's tests resolve against.
fn planAdjacentRuns(rows: []const u32, src: []u8, off: []u32) usize {
    const per = (rows.len + joinless_sources - 1) / joinless_sources;
    var at: u32 = 0;
    for (rows, src, off, 0..) |r, *s, *o, i| {
        if (i % per == 0) at = 0;
        s.* = @intCast(i / per);
        o.* = at;
        at += r;
    }
    return (rows.len + per - 1) / per;
}

/// The (output, row) a plan's (source, row) names.
fn resolvePlan(rows: []const u32, src: []const u8, off: []const u32, s: i32, row: i32) ?[2]u32 {
    if (s < 0 or row < 0) return null;
    const r_: u32 = @intCast(row);
    for (rows, src, off, 0..) |r, si, o, i| {
        if (si == s and r_ >= o and r_ < o + r) return .{ @intCast(i), r_ - o };
    }
    return null;
}

test "dsv41 experts: JOINLESS's merge at L1's counts: the 28 smallest of 51 outputs share one source, every assignment keeps its row" {
    const a = testing.allocator;
    try testing.expectEqual(@as(usize, @import("../../quant/exl3/stream/kernel_routes.zig").JoinlessCombine(TraceOps).sources), joinless_sources);
    // L1's layer at 16,384 tokens, modeled: 98,304 routed rows in 51 outputs (the deferred base call's 17
    // waves, then six transient calls' 34, each call's last wave ragged).
    const rows = [51]u32{
        3382, 3457, 3614, 3771, 3328, 3485, 3642, 3799, 3356, 3513, 3670, 3827, 3384, 3541, 3698, 3855, 3412,
        1237, 1298, 1359, 1120, 1181, 150,  1303, 1364, 1125, 1186, 1247, 240,  1369, 1130, 1191, 1252, 1313,
        330,  1135, 1196, 1257, 1318, 1379, 420,  1201, 1262, 1323, 1384, 510,  1206, 1267, 1328, 1389, 600,
    };
    var total: u32 = 0;
    for (rows) |r| total += r;
    try testing.expectEqual(@as(u32, 98_304), total);
    // The join order's assignments: a permutation (7919 is prime to 98,304).
    const pos = try a.alloc(u32, total);
    defer a.free(pos);
    for (pos, 0..) |*p, j| p.* = @intCast((j * 7919) % total);
    var order: [51]u32 = undefined;
    var src: [51]u8 = undefined;
    var off: [51]u32 = undefined;
    try testing.expectEqual(@as(usize, joinless_sources), planJoinless(&rows, &order, &src, &off));
    var n_merged: usize = 0;
    var merged_rows: u64 = 0;
    var max_merged: u32 = 0;
    var min_in_place: u32 = std.math.maxInt(u32);
    var next: u8 = 0;
    for (rows, src, off) |r, s, o| {
        if (s == joinless_sources - 1) {
            n_merged += 1;
            merged_rows += r;
            max_merged = @max(max_merged, r);
        } else {
            try testing.expectEqual(next, s);
            try testing.expectEqual(@as(u32, 0), o);
            next += 1;
            min_in_place = @min(min_in_place, r);
        }
    }
    try testing.expectEqual(@as(usize, 51 - 23), n_merged);
    try testing.expect(max_merged <= min_in_place);
    // The copy: 29,326 rows (0.60 GB at 20,480 B a row), at most (n - 23) / n of the rows by construction;
    // the adjacent runs (per 3) copied all 98,304.
    try testing.expectEqual(@as(u64, 29_326), merged_rows);
    try testing.expect(merged_rows * rows.len <= @as(u64, total) * (rows.len - 23));
    var a_src: [51]u8 = undefined;
    var a_off: [51]u32 = undefined;
    try testing.expectEqual(@as(usize, 17), planAdjacentRuns(&rows, &a_src, &a_off));
    // Every assignment reads its output's row, as under the adjacent runs.
    const loc = try a.alloc(i32, 2 * total);
    defer a.free(loc);
    const a_loc = try a.alloc(i32, 2 * total);
    defer a.free(a_loc);
    fillJoinlessLoc(&rows, &src, &off, pos, loc);
    fillJoinlessLoc(&rows, &a_src, &a_off, pos, a_loc);
    var j: usize = 0;
    for (rows, 0..) |r, i| for (0..r) |row| {
        const p = pos[j];
        j += 1;
        const want: [2]u32 = .{ @intCast(i), @intCast(row) };
        try testing.expectEqual(want, resolvePlan(&rows, &src, &off, loc[2 * p], loc[2 * p + 1]).?);
        try testing.expectEqual(want, resolvePlan(&rows, &a_src, &a_off, a_loc[2 * p], a_loc[2 * p + 1]).?);
    };
}

test "dsv41 experts: JOINLESS's merge copies at most (n - 23) / n of the routed rows at every n from 25 to 86, whatever the rows" {
    // The bill's merge term (`PrefillBill.joinedBytes`) charges (n_max - 23) / n_max of the routed rows at the most outputs a
    // call can make (86 at 16K); (n - 23) / n grows with n, so this bound at every n below it is what the bill relies on.
    var rng = std.Random.DefaultPrng.init(0xb0b_5e19);
    const r = rng.random();
    var rows: [86]u32 = undefined;
    var order: [86]u32 = undefined;
    var src: [86]u8 = undefined;
    var off: [86]u32 = undefined;
    var n: usize = joinless_sources + 1;
    while (n <= 86) : (n += 1) {
        for (0..5) |shape| {
            for (rows[0..n], 0..) |*x, i| x.* = switch (shape) {
                0 => 1, // all equal
                1 => @intCast(i + 1), // ascending
                2 => @intCast(n - i), // descending
                3 => if (i < n - joinless_sources + 1) 1 else 100_000, // many tiny, a few huge
                else => r.intRangeAtMost(u32, 1, 4096),
            };
            const n_src = planJoinless(rows[0..n], order[0..n], src[0..n], off[0..n]);
            try testing.expectEqual(@as(usize, joinless_sources), n_src);
            var total: u64 = 0;
            var copied: u64 = 0;
            var merged: usize = 0;
            for (rows[0..n], src[0..n]) |x, s_| {
                total += x;
                if (s_ == joinless_sources - 1) {
                    copied += x;
                    merged += 1;
                }
            }
            try testing.expectEqual(n - (joinless_sources - 1), merged);
            // copied / total <= (n - 23) / n, in integers.
            try testing.expect(copied * n <= total * (n - (joinless_sources - 1)));
        }
    }
}

test "dsv41 experts: JOINLESS's merge at 24 outputs or fewer: every output its own source, read in place, nothing copied" {
    const a = testing.allocator;
    var g = TraceOps.init(a);
    defer g.deinit();
    var sc: JoinlessScratch(u32) = .{};
    defer sc.deinit(a);
    for ([_]usize{ 1, 7, 24 }) |n_out| {
        var outs: std.ArrayList(u32) = .empty;
        defer outs.deinit(a);
        var n_rows: usize = 0;
        for (0..n_out) |i| {
            const r: c_int = @intCast(1 + (i * 5) % 9);
            try outs.append(a, try g.input(&.{ r, 64 }, .float32));
            n_rows += @intCast(r);
        }
        const before = try a.dupe(u32, outs.items);
        defer a.free(before);
        const pos = try a.alloc(u32, n_rows);
        defer a.free(pos);
        for (pos, 0..) |*p, j| p.* = @intCast(n_rows - 1 - j);
        const loc = try a.alloc(i32, 2 * n_rows);
        defer a.free(loc);
        const n0 = g.nodes.items.len;
        try testing.expectEqual(@as(u64, 0), try mergeJoinless(TraceOps, &g, a, &outs, pos, loc, &sc));
        try testing.expectEqual(n0, g.nodes.items.len);
        try testing.expectEqualSlices(u32, before, outs.items);
        var j: usize = 0;
        for (outs.items, 0..) |x, i| for (0..@intCast(g.shapeOf(x).dim(0))) |row| {
            const p = pos[j];
            j += 1;
            try testing.expectEqual(@as(i32, @intCast(i)), loc[2 * p]);
            try testing.expectEqual(@as(i32, @intCast(row)), loc[2 * p + 1]);
        };
    }
}

test "dsv41 experts: past 24 outputs the combine still reads each assignment's own row, in place or in the one merged source" {
    const a = testing.allocator;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 30;
    var src = try FakeSource.init(a, .{ .hidden = 64, .inter = 32, .n_experts = 30, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    g.record_host = true;
    const Math = WithPrefillRoutes(TraceOps, TraceMath, EncRoute);
    var rr = [_]EncRoute{.{}};
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rr }, &c);
    defer ex.deinit();
    // 20 tokens x top-6 over 30 experts in waves of 3 rows: about 40 outputs.
    const n: u32 = 20;
    const k: u32 = 6;
    var ids: [20 * 6]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(((i / k) * 5 + (i % k) * 7) % 30);
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    const first = g.nodes.items.len;
    const parts = try ex.at(0).routedParts(&g, try g.input(&.{ @intCast(n), 64 }, .bfloat16), try g.input(&.{ @intCast(n), @intCast(k) }, .int32));
    defer ex.at(0).releaseParts(&g);
    const jl = &ex.wide.jl;
    const n_out = jl.rows.items.len;
    try testing.expect(n_out > Ex.max_parts);
    try testing.expectEqual(@as(usize, Ex.max_parts), parts.outs.len);
    // One concatenate: the smallest n - 23 outputs' rows, the last source.
    var merged_rows: c_int = 0;
    var max_merged: u32 = 0;
    var min_in_place: u32 = std.math.maxInt(u32);
    for (jl.rows.items, jl.src.items) |r, s| {
        if (s == Ex.max_parts - 1) {
            merged_rows += @intCast(r);
            max_merged = @max(max_merged, r);
        } else min_in_place = @min(min_in_place, r);
    }
    try testing.expect(max_merged <= min_in_place);
    var n_cat: usize = 0;
    for (g.nodes.items[first..]) |nd| n_cat += @intFromBool(nd.op == .concat);
    try testing.expectEqual(@as(usize, 1), n_cat);
    try testing.expectEqual(merged_rows, g.shapeOf(parts.outs[Ex.max_parts - 1]).dim(0));
    // (profile builds) The merge's record for the prefill profile's group line: the outputs, the sources and the rows the
    // one concatenate copied; zeros elsewhere (compiled out).
    const ms = ex.at(0).lastMerge();
    if (comptime prof.enabled) {
        try testing.expectEqual(MergeStats{ .outputs = @intCast(n_out), .sources = Ex.max_parts, .copied_rows = @intCast(merged_rows) }, ms);
    } else try testing.expectEqual(MergeStats{}, ms);
    const call = for (&src.calls) |*cl| {
        if (cl.plan.n_ids > 0 and cl.plan.n_ids == ex.wide.distinct.items.len) break cl;
    } else return error.NoCall;
    const loc = ex.wide.loc.items;
    for (ids, 0..) |e, q| {
        const s_: usize = @intCast(loc[2 * q]);
        const r_: usize = @intCast(loc[2 * q + 1]);
        const bytes = g.hostBytesOf(parts.outs[s_]) orelse return error.NoHostBytes;
        const row = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(bytes)))[r_ * EncRoute.hidden ..][0..3];
        const ref = call.refs[@intCast(ex.wide.first.items[e])];
        try testing.expectEqual(@as(f32, @floatFromInt(q / k)), row[0]);
        try testing.expectEqual(@as(f32, @floatFromInt(ref.row)), row[1]);
        try testing.expectEqual(@as(f32, @floatFromInt(ex.banks[0][@backingInt(ref.bank)].?.gate.code)), row[2]);
    }
}

// Inside a GPU only: DSV41_PHASE0B_MLX=1 and DSV41_BANK (the real records). Device memory under 3 GB:
// 32 slot rows 0.43 GB, act 0.04 GB, the 51 outputs 0.50 GB, the three source sets at most 1.28 GB, the three
// combines and the shared rows 0.34 GB.
test "dsv41 smoke 0b: joinless merge: the combine over the minimal copy's sources equals the full concatenate's on real records, bit for bit" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse {
        std.debug.print("\njoinless merge smoke: DSV41_PHASE0B_MLX without DSV41_BANK (the real records): refused\n", .{});
        return error.TestUnexpectedResult;
    });
    const ks = sdk_ext.kernels.KernelSet(xk);
    const dkr = @import("../../quant/exl3/stream/kernel_routes.zig");
    const G = ops.MlxOps;
    const T = G.T;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{kd.message()});
        return e;
    };
    defer set.deinit();
    set.install(G, &g);
    defer ks.Set.uninstall(G, &g);
    // layer 0's first 16 experts, read by the stream into its 16 persistent MLX rows (the base bank)
    var bdiag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, testing.io, dir, expert_bank.dsv41, &bdiag) catch |e| {
        std.debug.print("bank refused: {s}\n", .{bdiag.message()});
        return e;
    };
    defer bank.deinit();
    const n_experts = 16;
    var base_rows: [40]u32 = @splat(0);
    base_rows[0] = n_experts;
    const st = try expert_stream.Stream.init(a, &bank, .{ .rows = &base_rows, .max_route_ids = n_experts, .transient_rows = n_experts, .slot_memory = .{ .mlx = s } });
    defer st.deinit();
    var ids: [n_experts]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    const route = try st.route(0, &ids, &.{});
    defer st.release(route);
    for (0..route.n_parts) |p| {
        try st.waitGu(route, @intCast(p));
        try st.waitDown(route, @intCast(p));
    }
    var refs: [max_route_ids]SlotRef = undefined;
    const rf = st.refsOf(route, &refs);
    try testing.expectEqual(@as(usize, n_experts), rf.len);
    for (rf) |r| try testing.expectEqual(BankKind.base, r.bank);
    const sb = st.bankArrays(0, .base) orelse return error.TestUnexpectedResult;
    const bk: BankArraysOf(T) = .{
        .gate = .{ .code = sb.gate.code, .rout = sb.gate.rout, .rin = sb.gate.rin },
        .up = .{ .code = sb.up.code, .rout = sb.up.rout, .rin = sb.up.rin },
        .down = .{ .code = sb.down.code, .rout = sb.down.rout, .rin = sb.down.rin },
    };
    // 4,096 tokens x top-6 in seven calls (a hot one, then six), waves of 2 experts: 51 outputs, L1's count.
    const n_tok = 4096;
    const n_ids = n_tok * 6;
    const call_experts = [_]usize{ 16, 16, 16, 14, 12, 11, 15 };
    var n_rows: [100]u32 = undefined;
    {
        var i: usize = 0;
        var sum: u32 = 0;
        for (call_experts, 0..) |ne, c| for (0..ne) |j| {
            n_rows[i] = (if (c == 0) @as(u32, 420) else 160) + @as(u32, @intCast((c * 7 + j * 5) % 9)) * 11;
            sum += n_rows[i];
            i += 1;
        };
        n_rows[i - 1] = n_rows[i - 1] + n_ids - sum;
    }
    const xs = try a.alloc(f32, n_tok * 5120);
    defer a.free(xs);
    var h: u64 = 0x9e3779b97f4a7c15;
    for (xs) |*v| {
        h = h *% 6364136223846793005 +% 1442695040888963407;
        const u_1: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 40)))) / 16777216.0;
        const u_2: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 16)) & 0xffffff)) / 16777216.0;
        v.* = (u_1 + u_2 - 1.0) * 2.0;
    }
    const act = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs), &.{ n_tok, 5120 }, .float32), .bfloat16);
    var dxp = try xq.DigXPrefill(G).init(a, &set.reg, .{ .wave = 2, .inflight = xq.PrefillShape.tier.inflight, .row_budget = xq.PrefillShape.tier.row_budget, .carry_rows = xq.PrefillShape.tier.carry_rows }, &kd);
    defer dxp.deinit(&g);
    var outs: std.ArrayList(T) = .empty;
    defer {
        for (outs.items) |x| g.release(x);
        outs.deinit(a);
    }
    var jpos: std.ArrayList(u32) = .empty;
    defer jpos.deinit(a);
    {
        var slot: std.ArrayList(u32) = .empty;
        defer slot.deinit(a);
        var act_row: std.ArrayList(u32) = .empty;
        defer act_row.deinit(a);
        var call_pos: std.ArrayList(u32) = .empty;
        defer call_pos.deinit(a);
        var wave_pos: std.ArrayList(u32) = .empty;
        defer wave_pos.deinit(a);
        // The assignments in a fixed permutation (7919 is prime to 24,576), each expert's run of them in turn.
        var q: usize = 0;
        var e: usize = 0;
        for (call_experts) |ne| {
            slot.clearRetainingCapacity();
            act_row.clearRetainingCapacity();
            call_pos.clearRetainingCapacity();
            for (0..ne) |j| {
                for (0..n_rows[e]) |_| {
                    const p: u32 = @intCast((q * 7919) % n_ids);
                    q += 1;
                    try slot.append(a, rf[j].row);
                    try act_row.append(a, p / 6);
                    try call_pos.append(a, p);
                }
                e += 1;
            }
            wave_pos.clearRetainingCapacity();
            try dxp.callParts(&g, act, .{ .slot = slot.items, .act_row = act_row.items }, bk, a, &outs, &wave_pos);
            for (wave_pos.items) |wp| try jpos.append(a, call_pos.items[wp]);
        }
    }
    try dxp.finish(&g);
    try g.evalAll(outs.items);
    const n_out = outs.items.len;
    try testing.expectEqual(@as(usize, 51), n_out);
    try testing.expectEqual(@as(usize, n_ids), jpos.items.len);
    const rows = try a.alloc(u32, n_out);
    defer a.free(rows);
    for (outs.items, rows) |x, *r| r.* = @intCast(g.shapeOf(x).dim(0));
    // 1. the full concatenate: one source, each assignment at its join row
    const full = g.keep(try g.concat(outs.items, 0));
    defer g.release(full);
    const inv = try a.alloc(u32, n_ids);
    defer a.free(inv);
    invertPositions(jpos.items, inv);
    const loc_full = try a.alloc(i32, 2 * n_ids);
    defer a.free(loc_full);
    for (inv, 0..) |r, p| {
        loc_full[2 * p] = 0;
        loc_full[2 * p + 1] = @intCast(r);
    }
    // 2. the adjacent runs (the merge before the minimal copy)
    const a_src = try a.alloc(u8, n_out);
    defer a.free(a_src);
    const a_off = try a.alloc(u32, n_out);
    defer a.free(a_off);
    const n_old = planAdjacentRuns(rows, a_src, a_off);
    var old: std.ArrayList(T) = .empty;
    defer {
        for (old.items) |x| g.release(x);
        old.deinit(a);
    }
    var copied_old: u64 = 0;
    {
        var o: usize = 0;
        while (o < n_out) {
            var end = o + 1;
            while (end < n_out and a_src[end] == a_src[o]) end += 1;
            if (end - o == 1) {
                try old.append(a, g.keep(outs.items[o]));
            } else {
                try old.append(a, g.keep(try g.concat(outs.items[o..end], 0)));
                for (rows[o..end]) |r| copied_old += r;
            }
            o = end;
        }
    }
    try testing.expectEqual(n_old, old.items.len);
    const loc_old = try a.alloc(i32, 2 * n_ids);
    defer a.free(loc_old);
    fillJoinlessLoc(rows, a_src, a_off, jpos.items, loc_old);
    // 3. the minimal copy: the lane's merge over its own handles of the same outputs
    var cur: std.ArrayList(T) = .empty;
    defer {
        for (cur.items) |x| g.release(x);
        cur.deinit(a);
    }
    for (outs.items) |x| try cur.append(a, g.keep(x));
    var sc: JoinlessScratch(T) = .{};
    defer sc.deinit(a);
    const loc_new = try a.alloc(i32, 2 * n_ids);
    defer a.free(loc_new);
    const copied_new = try mergeJoinless(G, &g, a, &cur, jpos.items, loc_new, &sc);
    try testing.expectEqual(@as(usize, joinless_sources), cur.items.len);
    // The three combines over the same weights and shared rows.
    const wv = try a.alloc(f32, n_ids);
    defer a.free(wv);
    for (wv, 0..) |*v, i| v.* = 0.05 + @as(f32, @floatFromInt((i * 2654435761) % 1000)) / 1100.0;
    const shv = try a.alloc(f32, n_tok * 5120);
    defer a.free(shv);
    for (shv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast((i * 40503) % 2001)) - 1000)) / 4096.0;
    const w_arr = try g.hostArray(std.mem.sliceAsBytes(wv), &.{ n_tok, 6 }, .float32);
    const sh_arr = try g.hostArray(std.mem.sliceAsBytes(shv), &.{ n_tok, 5120 }, .float32);
    const jl = try dkr.JoinlessCombine(G).init(&set.reg, &.derived, &kd);
    const locArr = struct {
        fn f(gg: *G, l: []const i32) !T {
            return gg.hostArray(std.mem.sliceAsBytes(l), &.{ n_tok, 6, 2 }, .int32);
        }
    }.f;
    const y_full = try jl.call(&g, &.{full}, try locArr(&g, loc_full), w_arr, sh_arr);
    const y_old = try jl.call(&g, old.items, try locArr(&g, loc_old), w_arr, sh_arr);
    const y_new = try jl.call(&g, cur.items, try locArr(&g, loc_new), w_arr, sh_arr);
    try g.evalAll(&.{ y_full, y_old, y_new });
    const words = n_tok * 5120;
    const buf = try a.alloc(f32, 3 * words);
    defer a.free(buf);
    const vf = try g.hostF32(y_full, buf[0..words]);
    const vo = try g.hostF32(y_old, buf[words .. 2 * words]);
    const vn = try g.hostF32(y_new, buf[2 * words ..]);
    var bad_old: usize = 0;
    var bad_new: usize = 0;
    var finite: usize = 0;
    for (vf, vo, vn) |x, y, z| {
        bad_old += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        bad_new += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(z)));
        finite += @intFromBool(std.math.isFinite(x));
    }
    var merged: usize = 0;
    for (sc.src.items) |sx| merged += @intFromBool(sx == joinless_sources - 1);
    std.debug.print("\nJOINLESS_MERGE_SMOKE {{\"outputs\":{d},\"rows\":{d},\"sources_new\":{d},\"sources_old\":{d},\"merged_outputs\":{d},\"copied_rows_new\":{d},\"copied_rows_old\":{d},\"copied_bytes_new\":{d},\"copied_bytes_old\":{d},\"words\":{d},\"finite\":{d},\"mismatch_new_vs_full\":{d},\"mismatch_old_vs_full\":{d}}}\n", .{ n_out, n_ids, cur.items.len, old.items.len, merged, copied_new, copied_old, copied_new * 5120 * 4, copied_old * 5120 * 4, words, finite, bad_new, bad_old });
    try testing.expectEqual(@as(usize, 0), bad_old);
    try testing.expectEqual(@as(usize, 0), bad_new);
    try testing.expectEqual(@as(usize, words), finite);
}

/// A decode math whose outputs carry, per row, its slot row and the arrays it read (host arrays the trace records):
/// the per-bank calls and the banked calls (packed ids) must hand every routed position the same words.
const BankEnc = struct {
    pub const has_banked = true;
    const w = 4;

    fn rowsOf(g: *TraceOps, x: u32) ![]const u32 {
        const b = g.hostBytesOf(x) orelse return error.NoHostBytes;
        return std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(b)));
    }

    fn floatsOf(g: *TraceOps, x: u32) ![]const f32 {
        const b = g.hostBytesOf(x) orelse return error.NoHostBytes;
        return std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(b)));
    }

    fn emit(g: *TraceOps, v: []const f32) !u32 {
        return g.hostArray(std.mem.sliceAsBytes(v), &.{ @intCast(v.len / w), w }, .float32);
    }

    pub fn gateUp(_: *const BankEnc, g: *TraceOps, _: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const r = try rowsOf(g, ids);
        for (r, 0..) |row, i| v[i * w ..][0..w].* = .{ @floatFromInt(row), @floatFromInt(gate.code), @floatFromInt(up.rin), 0 };
        return emit(g, v[0 .. r.len * w]);
    }

    pub fn down(_: *const BankEnc, g: *TraceOps, h: u32, _: u32, d: ProjOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const hv = try floatsOf(g, h);
        @memcpy(v[0..hv.len], hv);
        for (0..hv.len / w) |i| v[i * w + 3] = @floatFromInt(d.rout);
        return emit(g, v[0..hv.len]);
    }

    pub fn gateUpBanked(_: *const BankEnc, g: *TraceOps, _: u32, ids: u32, banks: *const [n_banks]BankArraysOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const r = try rowsOf(g, ids);
        for (r, 0..) |p, i| {
            const b = &banks[p >> 24];
            v[i * w ..][0..w].* = .{ @floatFromInt(p & 0xFFFFFF), @floatFromInt(b.gate.code), @floatFromInt(b.up.rin), 0 };
        }
        return emit(g, v[0 .. r.len * w]);
    }

    pub fn downBanked(_: *const BankEnc, g: *TraceOps, h: u32, ids: u32, banks: *const [n_banks]BankArraysOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const hv = try floatsOf(g, h);
        const r = try rowsOf(g, ids);
        @memcpy(v[0..hv.len], hv);
        for (r, 0..) |p, i| v[i * w + 3] = @floatFromInt(banks[p >> 24].down.rout);
        return emit(g, v[0..hv.len]);
    }
};

/// TraceMath with the banked texts (DEVROUTE's structural test): shapes only.
const TraceBankedMath = struct {
    pub const has_banked = true;
    inner: TraceMath,
    pub fn gateUp(self: *const TraceBankedMath, g: *ops.TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        return self.inner.gateUp(g, x, ids, gate, up);
    }
    pub fn down(self: *const TraceBankedMath, g: *ops.TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, d);
    }
    pub fn gateUpBanked(self: *const TraceBankedMath, g: *ops.TraceOps, x: u32, ids: u32, _: *const [n_banks]BankArraysOf(u32)) !u32 {
        if (g.shapeOf(x).d[1] != self.inner.hidden or g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inner.inter }, .float32);
    }
    pub fn downBanked(self: *const TraceBankedMath, g: *ops.TraceOps, h: u32, ids: u32, _: *const [n_banks]BankArraysOf(u32)) !u32 {
        if (g.shapeOf(h).d[1] != self.inner.inter or g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.inner.hidden }, .float32);
    }
};

test "dsv41 experts: DEVROUTE's invariant: before every decode route, each hit's LUT entry is the (bank, row) the host's plan serves it from" {
    // The device wave reads the LUT written after the layer's previous route; nothing else moves a layer's slots in
    // decode (the stream's policy changes only in route; failures fail the request). Over many routes with evictions:
    // every position the host's plan serves as a hit has LUT[expert] == its packed ref.
    const a = testing.allocator;
    const n_layers = 3;
    var src = try FakeSource.init(a, .{ .hidden = 64, .inter = 32, .n_experts = 40, .rows = &.{ 6, 6, 6 } });
    defer src.deinit();
    try src.grow(&.{ 12, 12, 12 });
    var lut: [n_layers][40]u32 = undefined;
    for (0..n_layers) |l| src.residentLut(@intCast(l), &lut[l]);
    var rng = std.Random.DefaultPrng.init(20261003);
    var hits_checked: usize = 0;
    var misses_seen: usize = 0;
    for (0..400) |call_i| {
        const l: u32 = @intCast(call_i % n_layers);
        var ids: [12]u16 = undefined;
        // 2 rows x top-6, a skewed draw (some repeats across calls, some new experts: evictions)
        for (&ids) |*d| d.* = @intCast(@min(39, rng.random().uintLessThan(u16, 18) + rng.random().uintLessThan(u16, 4) * rng.random().uintLessThan(u16, 6)));
        const call = try src.route(l, &ids, &.{});
        const sv = src.served(call);
        for (ids, sv.refs, sv.waves) |e, ref, w| {
            if (w != 0) {
                misses_seen += 1;
                continue;
            }
            try testing.expectEqual((@as(u32, @backingInt(ref.bank)) << 24) | ref.row, lut[l][e]);
            hits_checked += 1;
        }
        src.release(call);
        try src.flush();
        src.residentLut(l, &lut[l]);
    }
    try testing.expect(hits_checked > 2000 and misses_seen > 100);
}

test "dsv41 experts: DEVROUTE's join: each routed position's row is the last that carries it (miss parts after the device wave)" {
    const Ex = Experts(TraceOps, FakeSource, TraceMath);
    var order: [max_route_ids]u32 = undefined;
    // device rows 0..5 for positions 0..5, then miss parts for positions 2 and 4 (rows 6, 7)
    const n = Ex.joinOrder(&.{ 0, 1, 2, 3, 4, 5, 2, 4 }, 6, &order);
    try testing.expectEqual(@as(u32, 6), n);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 6, 3, 7, 5 }, order[0..6]);
    // stock: each position once
    _ = Ex.joinOrder(&.{ 3, 0, 1, 2 }, 4, &order);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 0 }, order[0..4]);
}

test "dsv41 experts: DEVROUTE: in decode the device hit wave is committed behind the barrier's arrays, before the wait; no host hit wave; the LUT flips per call" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 2);
    c.n_routed_experts = 16;
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = ExpertsWith(TraceOps, FakeSource, TraceBankedMath, .{});
    const math: TraceBankedMath = .{ .inner = .{ .hidden = 256, .inter = 128 } };
    // Refused without the banked route; off by default.
    try testing.expectError(error.DevrouteNeedsBanked, Ex.initWith(a, &g, &src, math, &c, .{ .devroute = true }));
    var ex = try Ex.initWith(a, &g, &src, math, &c, .{ .banked = true, .devroute = true });
    defer ex.deinit();
    try testing.expect(ex.devRoute());
    // A decode-width prompt call (before the grow) runs the stock body: no LUT yet.
    try testing.expectEqual(@as(usize, 0), ex.dev_lut.len);
    try ex.grow(&g, &.{ 8, 8 });
    try testing.expectEqual(@as(usize, 2), ex.dev_lut.len);
    src.log.clearRetainingCapacity();
    src.trace = &g;
    var script: Script = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
    g.host_values = script.values();
    const xf = try g.input(&.{ 2, 256 }, .bfloat16);
    const idx = try g.input(&.{ 2, 3 }, .int32);
    for (0..2) |call_i| {
        const before = ex.dev_parity[0];
        const start = g.nodes.items.len;
        const out = try ex.at(0).routedHoist(&g, xf, idx, &.{});
        try testing.expect(g.shapeOf(out).eql(ops.Shape.of(&.{ 2, 3, 256 })));
        var commits_before_read: usize = 0;
        var read_seen = false;
        var commits_after_route: usize = 0;
        var r_at: usize = 0; // the call's route event's node index
        for (src.log.items) |ev| if (ev.kind == .route) {
            r_at = ev.at;
        };
        for (g.nodes.items[start..], start..) |nd, i| switch (nd.op) {
            .async_eval => {
                if (!read_seen) commits_before_read += 1;
                if (i >= r_at) commits_after_route += 1;
            },
            .host_read => read_seen = true,
            else => {},
        };
        // the barrier's arrays, then the device wave; after the route only the miss parts' commits (call 1: two parts
        // of gate/up + down = 4; call 2: one part = 2)
        try testing.expectEqual(@as(usize, 2), commits_before_read);
        try testing.expectEqual(@as(usize, if (call_i == 0) 4 else 2), commits_after_route);
        try testing.expect(ex.dev_parity[0] != before);
        // the LUT the next call reads is the policy after this route
        var want: [16]u32 = undefined;
        src.residentLut(0, &want);
        const got = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[0][ex.dev_parity[0]]))));
        try testing.expectEqualSlices(u32, &want, got);
    }
    try ex.flush();
}

/// One run of two decode calls under DEVROUTE and / or HOIST_FIRST (stack4's routed-path pair), recorded for comparison:
/// the non-commit nodes, and per call the classes of the commits before the routing barrier's read (B the barrier's
/// arrays, D the device hit wave, H the hoist) and the commits after the route.
const DevHoistRun = struct {
    ops: std.ArrayList(ops.Op) = .empty,
    shapes: std.ArrayList(ops.Shape) = .empty,
    pre: [2][8]u8 = undefined,
    n_pre: [2]usize = .{ 0, 0 },
    post_route: [2]usize = .{ 0, 0 },
    hoist_after_read: [2]usize = .{ 0, 0 },
    lut_ok: [2]bool = .{ false, false },
    flipped: [2]bool = .{ false, false },
    bound_dev: bool = false,
    bound_hoist: bool = false,

    fn deinit(r: *DevHoistRun, a: std.mem.Allocator) void {
        r.ops.deinit(a);
        r.shapes.deinit(a);
    }

    fn preOf(r: *const DevHoistRun, call: usize) []const u8 {
        return r.pre[call][0..r.n_pre[call]];
    }

    fn run(comptime R: Routes, a: std.mem.Allocator, devroute: bool, hoist_first: bool, out: *DevHoistRun) !void {
        var c = testConfig(256, 128, 2);
        c.n_routed_experts = 16;
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
        defer src.deinit();
        if (R.lookahead) src.selector = try expert_lookahead.Selector.init(a, 16, 6, std.math.inf(f32), 2);
        var g = TraceOps.init(a);
        defer g.deinit();
        g.record_host = true;
        const Ex = ExpertsWith(TraceOps, FakeSource, TraceBankedMath, R);
        const math: TraceBankedMath = .{ .inner = .{ .hidden = 256, .inter = 128 } };
        var gates: [2]Ex.Gate = undefined;
        for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 16, 256 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
        var ex = try Ex.initWith(a, &g, &src, math, &c, .{ .gates = if (R.lookahead) &gates else &.{}, .banked = true, .devroute = devroute, .hoist_first = hoist_first });
        defer ex.deinit();
        out.bound_dev = ex.devRoute();
        out.bound_hoist = ex.hoistFirst();
        try ex.grow(&g, &.{ 8, 8 });
        src.log.clearRetainingCapacity();
        src.trace = &g;
        // Call 1: six misses (two parts); call 2: three hits, one part.
        var vals: HoistRun.Values = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
        g.host_values = vals.values();
        const xf = try g.input(&.{ 2, 256 }, .bfloat16);
        const idx = try g.input(&.{ 2, 3 }, .int32);
        const tail = [_]u32{ try g.input(&.{ 2, 256 }, .float32), try g.input(&.{ 2, 3 }, .float32) };
        const first = g.nodes.items.len;
        for (0..2) |call| {
            const parity0 = if (devroute) ex.dev_parity[0] else 0;
            const n0 = g.nodes.items.len;
            const c0 = g.commit_spans.items.len;
            const out_arr = try ex.at(0).routedHoist(&g, xf, idx, &tail);
            if (!g.shapeOf(out_arr).eql(ops.Shape.of(&.{ 2, 3, 256 }))) return error.JoinShape;
            // The call's route event (the last in the source's log) and its first host read.
            var r_at: usize = 0;
            for (src.log.items) |ev| if (ev.kind == .route) {
                r_at = ev.at;
            };
            var read_at: usize = g.nodes.items.len;
            for (g.nodes.items[n0..], n0..) |nd, i| if (nd.op == .host_read) {
                read_at = i;
                break;
            };
            var k = c0;
            for (g.nodes.items[n0..], n0..) |nd, i| if (nd.op == .async_eval) {
                const span = g.commit_spans.items[k];
                k += 1;
                const xs = g.committed.items[span[0]..span[1]];
                const cls: u8 = if (std.mem.indexOfScalar(u32, xs, idx) != null) 'B' else if (std.mem.indexOfScalar(u32, xs, tail[0]) != null) 'H' else 'D';
                if (i < read_at) {
                    out.pre[call][out.n_pre[call]] = cls;
                    out.n_pre[call] += 1;
                } else if (cls == 'H') out.hoist_after_read[call] += 1;
                if (i >= r_at and cls != 'H') out.post_route[call] += 1;
            };
            if (devroute) {
                out.flipped[call] = ex.dev_parity[0] != parity0;
                var want: [16]u32 = undefined;
                src.residentLut(0, &want);
                const got = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[0][ex.dev_parity[0]]))));
                out.lut_ok[call] = std.mem.eql(u32, &want, got);
            }
        }
        for (g.nodes.items[first..]) |nd| if (nd.op != .async_eval) {
            try out.ops.append(a, nd.op);
            try out.shapes.append(a, nd.shape);
        };
        try ex.flush();
    }
};

test "dsv41 experts: stack4's routed pair: DEVROUTE with HOIST_FIRST commits the barrier, the device wave, then the hoist, all before the wait; only commits move; the LUT still flips to the post-route policy" {
    const a = testing.allocator;
    inline for (.{ Routes{}, Routes{ .gated = true }, Routes{ .gated = true, .lookahead = true } }) |R| {
        var dev: DevHoistRun = .{};
        defer dev.deinit(a);
        var both: DevHoistRun = .{};
        defer both.deinit(a);
        try DevHoistRun.run(R, a, true, false, &dev);
        try DevHoistRun.run(R, a, true, true, &both);
        try testing.expect(dev.bound_dev and !dev.bound_hoist and both.bound_dev and both.bound_hoist);
        // Exact relative to DEVROUTE alone: every non-commit node in the same order and shape.
        try testing.expectEqualSlices(ops.Op, dev.ops.items, both.ops.items);
        for (dev.shapes.items, both.shapes.items) |x, y| try testing.expect(x.eql(y));
        for (0..2) |call| {
            // DEVROUTE alone: the barrier's arrays, then the device wave, then the wait (its hoist after the hit point).
            try testing.expectEqualStrings("BD", dev.preOf(call));
            // With HOIST_FIRST: the barrier, the device wave, the barrier re-committed (a no-op), the hoist: the hoist is
            // behind the device wave and before the wait, and never committed again after it.
            try testing.expectEqualStrings("BDBH", both.preOf(call));
            try testing.expectEqual(@as(usize, 0), both.hoist_after_read[call]);
            // After the route: only the miss parts' commits, the same with and without the hoist moved.
            try testing.expectEqual(dev.post_route[call], both.post_route[call]);
            // The LUT pair flips per call and holds the policy after this route.
            try testing.expect(both.flipped[call] and both.lut_ok[call] and dev.flipped[call] and dev.lut_ok[call]);
        }
        try testing.expectEqual(@as(usize, 1), dev.hoist_after_read[0]);
    }
}

test "dsv41 experts: the banked waves (ROUTED_BANKED) hand every routed position the per-bank route's words, one group per wave" {
    const a = testing.allocator;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 30;
    var src = try FakeSource.init(a, .{ .hidden = 64, .inter = 32, .n_experts = 30, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = ExpertsWith(TraceOps, FakeSource, BankEnc, .{});
    // One hook, both bindings (the arrays bound once): the option binds the banked stages at construction.
    var banked = try Ex.initWith(a, &g, &src, .{}, &c, .{ .banked = true });
    defer banked.deinit();
    try testing.expect(banked.gate_up_wave == &Ex.gateUpWaveBanked and banked.down_wave == &Ex.downWaveBanked);
    var grouped = banked;
    grouped.gate_up_wave = Ex.gateUpWaveGrouped;
    grouped.down_wave = Ex.downWaveGrouped;
    // The ext bank is unbound: the banked arrays fill it with another bank's (never indexed).
    try testing.expect(banked.banks[0][@backingInt(BankKind.ext)] == null);
    try testing.expect(banked.banks[0][@backingInt(BankKind.base)].?.gate.code != banked.banks[0][@backingInt(BankKind.transient)].?.gate.code);
    // One token, top-6, both banks in each wave.
    const refs = [_]SlotRef{ .{ .bank = .base, .row = 3 }, .{ .bank = .transient, .row = 5 }, .{ .bank = .base, .row = 7 }, .{ .bank = .transient, .row = 0 }, .{ .bank = .base, .row = 1 }, .{ .bank = .transient, .row = 2 } };
    const waves = [_]u8{ 0, 0, 1, 1, 0, 1 };
    const sv: Served = .{ .refs = &refs, .waves = &waves, .n_parts = 1 };
    const xf = try g.input(&.{ 1, 64 }, .bfloat16);
    var words: [2][6][BankEnc.w]f32 = undefined;
    for ([_]*Ex{ &grouped, &banked }, 0..) |ex, arm| {
        var acc: Ex.Acc = .{};
        for (0..2) |wv| {
            const wave = try ex.gate_up_wave(ex, &g, 0, xf, 6, sv, @intCast(wv), null);
            try testing.expectEqual(@as(usize, if (arm == 0) 2 else 1), wave.n);
            _ = try ex.down_wave(ex, &g, 0, &wave, &acc, null);
        }
        try testing.expectEqual(@as(usize, if (arm == 0) 4 else 2), acc.n_outs);
        var j: usize = 0;
        for (acc.outs[0..acc.n_outs]) |o| {
            const v = try BankEnc.floatsOf(&g, o);
            for (0..v.len / BankEnc.w) |r| {
                words[arm][acc.pos[j]] = v[r * BankEnc.w ..][0..BankEnc.w].*;
                j += 1;
            }
        }
        try testing.expectEqual(@as(usize, 6), j);
    }
    for (words[0], words[1], refs) |x, y, ref| {
        try testing.expectEqual(x, y);
        try testing.expectEqual(@as(f32, @floatFromInt(ref.row)), x[0]);
        try testing.expectEqual(@as(f32, @floatFromInt(grouped.banks[0][@backingInt(ref.bank)].?.gate.code)), x[1]);
    }
    // A math without the banked route refuses the option at construction.
    const Plain = ExpertsWith(TraceOps, FakeSource, TraceMath, .{});
    try testing.expectError(error.BankedNotInMath, Plain.initWith(a, &g, &src, .{ .hidden = 64, .inter = 32 }, &c, .{ .banked = true }));
}
