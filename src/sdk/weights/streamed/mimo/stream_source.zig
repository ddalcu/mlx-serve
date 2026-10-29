//! The shared SDK stream's source bridge over any bank module (`For`): the stream built from the arch's rows and the
//! construction routes (the read pool, the event gates, the lookahead, P7's read schedule), and the `StreamSource` the
//! shared DeepSeek scheduler (`deepseek_v41_experts.ExpertsWith`) runs over: absolute model layers, the route ring's calls,
//! the read schedule's routes and the bank's slot bindings. MiMo's instance is stream.zig's (its MXFP4 bank, GatherQmm's
//! {w, scales}); GLM-5.3-Flash's is glm5_next/experts.zig's (its NVFP4 bank, {w, scales, global_scale}). The code below
//! is MiMo's bridge as it was, parameterized by the bank module and its projection arrays.
const std = @import("std");
const sdk_ext = @import("../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const expert_policy = sdk_ext.expert.policy;
const profile = @import("profile.zig");
const route_trace = @import("route_trace.zig");
const prefill_timeline = @import("prefill_timeline.zig");
const gpu_warm = @import("gpu_warm.zig");
const diag = @import("diag.zig");
const shared = @import("stream.zig");
const ReadSchedule = shared.ReadSchedule;
const AutoSchedule = shared.AutoSchedule;
const max_read_schedule_threshold = shared.max_read_schedule_threshold;
const io_workers = shared.io_workers;
const io_tickets = shared.io_tickets;
const io_tickets_early = shared.io_tickets_early;
const records_per_part = shared.records_per_part;
const wide_depth = shared.wide_depth;
const transient_rows = shared.transient_rows;

/// What an arch's bank hands the bridge besides its module.
pub const Params = struct {
    /// The projection arrays a slot bank binds (`ExpertsWith` holds them equal to its math's `Projection`).
    Arrays: fn (type) type,
    /// The read pool's staging per worker: at least the bank's widest gate/up or down span, page-rounded, plus a page
    /// (`StreamOf.stagingBytes`).
    staging_bytes: u64,
    /// Profile builds: the route trace's type (route_trace.zig `TraceOf`: u8 ids for at most 256 experts, MiMo's).
    Trace: type = route_trace.Trace,
};

/// The bridge over `bank_mod` (an SDK bank module: `bank_mod.SDK` for `StreamOf`, its `Layer`, `n_components`, `mlxDtype`,
/// `BankArrays` and `BankArraysOf`).
pub fn For(comptime bank_mod: type, comptime params: Params) type {
    return struct {
        pub const S = sdk_ext.expert.stream.StreamOf(bank_mod.SDK, false);
        pub const Stream = S.Stream;
        pub const Options = S.Options;
        pub const SlotMemory = S.SlotMemory;
        pub const LayerSlotBank = S.LayerSlotBank;
        pub const Route = S.Route;
        pub const Part = S.Part;
        pub const BankKind = S.BankKind;
        pub const SlotRef = S.SlotRef;
        pub const Error = S.Error;
        pub const Stats = S.Stats;
        pub const Gates = S.Gates;
        pub const max_route_ids = S.max_route_ids;
        pub const uses_reader = S.uses_reader;
        pub const BankArrays = bank_mod.BankArrays;
        pub const BankArraysOf = bank_mod.BankArraysOf;

        pub const lookahead: S.Lookahead = .{ .k = 8, .tau = std.math.inf(f32), .budget = 2, .chunks = 4, .idle_busy = 0, .preread = true };

        /// Persistent rows are the caller's admitted routed-layer rows (47 for MiMo).
        /// This adds two 48-row prefill windows, not a second cache or old slot bank.
        pub fn options(rows: []const u32, memory: SlotMemory) Options {
            return .{
                .rows = rows,
                .max_route_ids = max_route_ids,
                .transient_rows = transient_rows,
                .records_per_part = records_per_part,
                .wide_depth = wide_depth,
                .pool = .{ .workers = io_workers, .staging_bytes = params.staging_bytes, .tickets = io_tickets },
                .slot_memory = memory,
            };
        }

        /// The read schedule's stream options (construction): `early` installs the stream's overlap schedule on the wider ticket
        /// ring, its read-ahead admissions pinned to the stock ring's bound (`ahead_cap`), so they stay the control's. `auto`
        /// builds the same stream as `early` (each request's start sets its read-ahead post: `StreamSource.chooseReadSchedule`).
        pub fn applyReadSchedule(opts: *Options, schedule: ReadSchedule) !void {
            switch (schedule) {
                .current => {},
                .early, .early_whole, .early_whole_trim, .auto => {
                    opts.ahead_cap = (opts.pool.tickets - 2 * sdk_ext.expert.io.max_items) / 2;
                    opts.prompt_reads = .overlap;
                    opts.prompt_ahead = aheadPost(schedule);
                    opts.pool.tickets = io_tickets_early;
                },
                .invalid => return error.MimoPrefillReadSchedule,
            }
        }

        /// The stream's read-ahead post of an `overlap` schedule (`applyReadSchedule`, `installReadSchedule`, `scheduleAuto`):
        /// `early` and `auto` build the head (an `auto` request's start then sets its entry's: head or whole).
        pub fn aheadPost(schedule: ReadSchedule) S.AheadPost {
            return switch (schedule) {
                .early_whole => .whole,
                .early_whole_trim => .whole_trim,
                else => .head,
            };
        }

        /// One explicit native gated/lookahead construction route. The genuine phase
        /// enables prediction and pre-read only after the decode boundary.
        pub fn eventOptions(rows: []const u32, memory: SlotMemory, event: sdk_ext.expert.event.Event) Options {
            var opts = options(rows, memory);
            opts.event = .{ .backend = .{ .metal = event.object }, .watchdog_ms = 10_000 };
            opts.lookahead = lookahead;
            return opts;
        }

        /// The wave view of the SDK's existing route plan, without another load plan
        /// or a matrix of copied part loads. Hits are wave 0; part p is wave p + 1.
        pub fn wavesOf(r: *const Route, out: []u8) void {
            var slots: [max_route_ids]u32 = undefined;
            var waves: [max_route_ids]u8 = undefined;
            var n: usize = 0;
            for (r.hit_slots[0..r.plan.n_hits]) |slot| {
                slots[n] = slot;
                waves[n] = 0;
                n += 1;
            }
            for (r.partsOf(), 1..) |part, wave| {
                for (r.order[part.first..][0..part.n]) |li| {
                    slots[n] = r.plan.loads[li].slot;
                    waves[n] = @intCast(wave);
                    n += 1;
                }
            }
            for (r.plan.slotsOf(), out) |slot, *wave| wave.* = waves[std.mem.indexOfScalar(u32, slots[0..n], slot).?];
        }

        pub fn traceBank(g: anytype, layer: *const bank_mod.Layer, rows: u32) !BankArraysOf(@TypeOf(g.*).T) {
            var arrays: [bank_mod.n_components]@TypeOf(g.*).T = undefined;
            for (&arrays, layer.segments) |*array, seg| {
                var shape: [4]c_int = undefined;
                shape[0] = @intCast(rows);
                for (seg.shape[0..seg.rank], 1..) |d, i| shape[i] = @intCast(d);
                array.* = try g.input(shape[0 .. seg.rank + 1], bank_mod.mlxDtype(seg.dtype));
            }
            return .{
                .gate = .{ .w = arrays[0], .scales = arrays[1] },
                .up = .{ .w = arrays[2], .scales = arrays[3] },
                .down = .{ .w = arrays[4], .scales = arrays[5] },
            };
        }

        /// Borrowed source over an installed SDK Stream. Its calls mirror that
        /// stream's route ring. It owns no rows, policy, pool, fd or MLX arrays.
        pub const StreamSource = struct {
            stream: *Stream,
            calls: [n_routes]Call = @splat(.{}),
            /// Profile builds only (`mimo-route-trace-v2`, `params.Trace`): the request's route trace, when one is armed.
            trace: if (profile.enabled) ?*params.Trace else void = if (profile.enabled) null else {},
            /// `MIMO_DIAG=missfree` (profile builds; the CPU tests): the decode phase's misses served from resident slots
            /// (`diag.MissFree`). Owned by the source that armed it. `void` in a served build.
            missfree: if (diag.compiled) ?*diag.MissFree else void = if (diag.compiled) null else {},
            /// P7 (`ReadSchedule`): the prompt routes, their release and the executor's read-ahead await, bound once
            /// (`installReadSchedule`); `early` takes a prompt call's groups from its hot order (`hotOrder`).
            prompt_route: *const fn (*StreamSource, u32, []const u16, []const f32) Error!*Route = routeStock,
            prompt_release: *const fn (*StreamSource, *Route) void = releaseStock,
            prompt_await: *const fn (*StreamSource, u32) Error!void = awaitStock,
            early: bool = false,
            pending: []const u16 = &.{},
            pending_layer: u32 = 0,
            /// X20: a request's read schedule, bound once (`installReadSchedule`): a named schedule is every request's
            /// (`scheduleNamed`: nothing is set per request); `auto` chooses from the request's prompt length (`scheduleAuto`:
            /// `AutoSchedule`, the stream's read-ahead post set once, at the request's start). `chooseReadSchedule` calls it.
            request_schedule: *const fn (*StreamSource, u64) anyerror!ReadSchedule = scheduleNamed,
            installed_schedule: ReadSchedule = .current,
            auto_schedule: AutoSchedule = .{},
            /// `gpu_keep_warm: "waits"`: the warm engine's arm, bound once at construction (null = the control). `gate` hands it the value the call's last gated wait needs (both phases);
            /// `awaitReadAhead` holds it for the host's read-ahead wait.
            warm_arm: ?gpu_warm.Arm = null,

            pub const Arrays = params.Arrays;
            pub const uses_reader = S.uses_reader;
            pub const caps = blk: {
                var c = S.source_caps;
                c.construction_reset = true;
                break :blk c;
            };
            pub const n_routes = @typeInfo(@FieldType(Stream, "routes")).array.len;
            pub const Call = struct {
                route: ?*Route = null,
                n_ids: u32 = 0,
                refs: [max_route_ids]SlotRef = undefined,
                waves: [max_route_ids]u8 = undefined,
            };

            pub fn init(stream: *Stream) StreamSource {
                return .{ .stream = stream };
            }

            /// The shared wide feed's hottest-first order (`hot_first`), right after this call's seed.
            pub fn hotOrder(self: *StreamSource, layer: u32, distinct: []u16, counts: []const u32) void {
                // In decode no seed ranked this call (`seedPrefill`): the stock count order.
                if (self.decoding()) return orderCounts(self, layer, distinct, counts);
                orderCounts(self, layer, distinct, counts);
                // P7 `early`: the call's groups as the executor routes them (its first route plans them all).
                if (self.early) {
                    self.pending = distinct;
                    self.pending_layer = self.relativeLayer(layer);
                }
            }

            /// P7: binds the read schedule's routes, once at construction; `early` needs the stream built with its options
            /// (`applyReadSchedule`) and a prompt call whose groups fit the stream's route ring. X20 `auto` binds `early`'s
            /// routes (both of its entries run them) and the per-request choice at `threshold` (1..`max_read_schedule_threshold`;
            /// `threshold` is read for `auto` only).
            pub fn installReadSchedule(self: *StreamSource, schedule: ReadSchedule, threshold: u32) !void {
                switch (schedule) {
                    .current => {
                        if (self.stream.prompt_reads != .stock) return error.MimoPrefillReadSchedule;
                        self.prompt_route = routeStock;
                        self.prompt_release = releaseStock;
                        self.prompt_await = awaitStock;
                        self.early = false;
                    },
                    .early, .early_whole, .early_whole_trim, .auto => {
                        if (self.stream.prompt_reads != .overlap) return error.MimoPrefillReadSchedule;
                        // `auto` sets the post per request (`scheduleAuto`); a named schedule's is the stream's own.
                        if (schedule != .auto and self.stream.prompt_ahead != aheadPost(schedule)) return error.MimoPrefillReadSchedule;
                        if (schedule == .auto and (threshold == 0 or threshold > max_read_schedule_threshold)) return error.MimoPrefillReadSchedule;
                        // The executor's groups: `max_route_ids` wide, all of a layer's experts at most; one route each.
                        if ((self.stream.bank.n_experts + self.stream.max_route_ids - 1) / self.stream.max_route_ids > S.route_capacity)
                            return error.MimoPrefillReadSchedule;
                        self.prompt_route = routeEarly;
                        self.prompt_release = releaseEarly;
                        self.prompt_await = awaitNone;
                        self.early = true;
                    },
                    .invalid => return error.MimoPrefillReadSchedule,
                }
                self.installed_schedule = schedule;
                if (schedule == .auto) {
                    self.auto_schedule = .{ .threshold = threshold };
                    self.request_schedule = scheduleAuto;
                } else self.request_schedule = scheduleNamed;
            }

            /// X20: a request's read schedule, chosen once at its start (`Source.chooseReadSchedule`, after `beginPrefill`: nothing
            /// of a prompt pass is live) from its prompt length, through the route `installReadSchedule` bound. Every prompt call
            /// of the request (its chunks, a same-request extend) runs it; the next request's start chooses again.
            pub fn chooseReadSchedule(self: *StreamSource, prompt_tokens: u64) !ReadSchedule {
                return self.request_schedule(self, prompt_tokens);
            }

            pub fn scheduleNamed(self: *StreamSource, _: u64) anyerror!ReadSchedule {
                return self.installed_schedule;
            }

            pub fn scheduleAuto(self: *StreamSource, prompt_tokens: u64) anyerror!ReadSchedule {
                const schedule = self.auto_schedule.of(prompt_tokens);
                try self.stream.setPromptAhead(aheadPost(schedule));
                return schedule;
            }

            pub fn routeStock(self: *StreamSource, rel: u32, ids: []const u16, scores: []const f32) Error!*Route {
                return self.stream.route(rel, ids, scores);
            }

            /// `early`: a hot-ordered prompt call's first group plans the whole call (`Stream.planCall`); its groups then route
            /// in order (`Stream.activateP7`). Any other route (a decode-lane call, a parity check) is the stock route.
            pub fn routeEarly(self: *StreamSource, rel: u32, ids: []const u16, scores: []const f32) Error!*Route {
                const st = self.stream;
                if (st.phase == .prefill) {
                    if (st.p7.live and st.p7.layer == rel) return st.activateP7(ids);
                    if (self.pending.len > 0 and self.pending_layer == rel and ids.ptr == self.pending.ptr) {
                        const call = self.pending;
                        self.pending = &.{};
                        try st.planCall(rel, call, st.max_route_ids);
                        return st.activateP7(ids);
                    }
                }
                return st.route(rel, ids, scores);
            }

            pub fn releaseStock(self: *StreamSource, r: *Route) void {
                self.stream.release(r);
            }

            pub fn releaseEarly(self: *StreamSource, r: *Route) void {
                const st = self.stream;
                if (st.p7.live and st.p7.layer == r.layer) st.releaseP7(r) else st.release(r);
            }

            pub fn awaitStock(self: *StreamSource, rel: u32) Error!void {
                return self.stream.awaitReadAhead(rel);
            }

            /// `early`: the executor's await before the seed lands nothing; the plans land what they need (`Stream.planCall`).
            pub fn awaitNone(_: *StreamSource, _: u32) Error!void {}

            /// The stock order: count descending, ties by id (the seed's `rankHottest`).
            pub fn orderCounts(_: *StreamSource, _: u32, distinct: []u16, counts: []const u32) void {
                std.sort.pdq(u16, distinct, counts, struct {
                    fn lt(c: []const u32, x: u16, y: u16) bool {
                        return if (c[x] != c[y]) c[x] > c[y] else x < y;
                    }
                }.lt);
            }

            pub fn decoding(self: *const StreamSource) bool {
                return self.stream.phase == .decode;
            }

            // ── HITPASS (DEVROUTE, design-decode-fusion 2d) ──
            /// Absolute layer `layer`'s resident map as the device reads it: `out[e]` = the packed (bank << 24 | row) of expert
            /// e's persistent slot, else 0 (a miss: its device row reads base row 0 and is never joined). DSV's `lutOf` over the
            /// relative layer's policy; the dense layers (below `first_moe_layer`) have no policy and map every expert to 0.
            /// The scheduler writes it only after a route (the policy changes only in `route`), never per dispatch.
            pub fn residentLut(self: *const StreamSource, layer: u32, out: []u32) void {
                if (layer < self.stream.bank.first_moe_layer) return @memset(out, 0);
                if (comptime diag.compiled) if (self.missfree) |mf| if (self.decoding()) return self.missFreeLut(mf, layer, out);
                const rel = self.relativeLayer(layer);
                @import("../expert/stream.zig").lutOf(&self.stream.layers[rel].policy, self.stream, rel, out);
            }

            // ── end HITPASS ──

            pub fn relativeLayer(self: *const StreamSource, layer: u32) u32 {
                return layer - self.stream.bank.first_moe_layer;
            }

            pub fn callOf(self: *StreamSource, r: *Route) *Call {
                const i = (@intFromPtr(r) - @intFromPtr(&self.stream.routes[0])) / @sizeOf(Route);
                return &self.calls[i];
            }

            /// Stream.route submits both GU and down asynchronously. Neither wait
            /// runs here: the shared scheduler chooses when to consume each part.
            pub fn route(self: *StreamSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
                if (comptime diag.compiled) if (self.missfree) |mf| if (self.decoding()) return self.routeMissFree(mf, layer, ids);
                const r = try self.prompt_route(self, self.relativeLayer(layer), ids, scores);
                if (comptime profile.enabled) if (self.decoding()) {
                    if (self.trace) |t| t.decodeRoute(self.relativeLayer(layer), ids, &r.plan, scores);
                };
                const call = self.callOf(r);
                call.* = .{ .route = r, .n_ids = @intCast(ids.len) };
                _ = self.stream.refsOf(r, &call.refs);
                wavesOf(r, call.waves[0..ids.len]);
                return call;
            }

            pub fn served(_: *StreamSource, call: *const Call) sdk_ext.expert.Served {
                return .{ .refs = call.refs[0..call.n_ids], .waves = call.waves[0..call.n_ids], .n_parts = call.route.?.n_parts };
            }

            pub fn waitGu(self: *StreamSource, call: *Call, part: u32) Error!void {
                return self.stream.waitGu(call.route.?, part);
            }

            pub fn waitDown(self: *StreamSource, call: *Call, part: u32) Error!void {
                return self.stream.waitDown(call.route.?, part);
            }

            pub fn release(self: *StreamSource, call: *Call) void {
                self.prompt_release(self, call.route.?);
            }

            /// The caller has evaluated every released route's consumers first.
            pub fn flush(self: *StreamSource) Error!void {
                return self.stream.flush();
            }

            pub fn policy(self: *StreamSource, layer: u32) *expert_policy.LayerPolicy {
                return &self.stream.layers[self.relativeLayer(layer)].policy;
            }

            pub fn slotCapacity(self: *const StreamSource, layer: u32) u32 {
                return self.stream.layers[self.relativeLayer(layer)].policy.capacity;
            }

            pub fn seedPrefill(self: *StreamSource, layer: u32, ids: []const u16) !void {
                // A decode-phase wide call routes against the decode residency: the residency seed is the prompt's (the stream
                // refuses it in decode, `NotPrefill`). A phase route. (The MTP verify, M <= 4, takes the decode lane since
                // `Source.decodeLane`; this keeps any decode-phase M >= 5 call off the refusal.)
                if (self.decoding()) return;
                if (comptime profile.enabled) if (self.trace) |t| t.seed(self.relativeLayer(layer), ids);
                return self.stream.seedPrefill(self.relativeLayer(layer), ids);
            }

            pub fn seedRanks(self: *const StreamSource, layer: u32) u32 {
                return self.stream.seedRanks(self.relativeLayer(layer));
            }

            pub fn holdBase(self: *StreamSource, call: *Call) !void {
                return self.stream.holdBase(call.route.?);
            }

            pub fn releaseHeld(self: *StreamSource) void {
                self.stream.releaseHeld();
            }

            pub fn wideDepth(self: *const StreamSource) u8 {
                return self.stream.wide_depth;
            }

            /// Row vectors always use SDK routed-layer order, without a dense layer 0.
            pub fn grow(self: *StreamSource, rows: []const u32) !void {
                try self.stream.grow(rows);
                if (comptime diag.compiled) if (self.missfree) |mf| for (self.stream.layers, 0..) |*ls, l| try mf.fill(@intCast(l), &ls.policy);
            }

            // ── MIMO_DIAG=missfree (profile builds and the CPU tests only; `diag.zig`) ──
            /// The decode route with every miss substituted by a resident expert of the layer: the plan has hits only (no load,
            /// no part, no gate, no admission), and no scores reach the stream (no speculation; the pre-read finds no certain
            /// miss). The call's refs are the substitutes' slots, as the device LUT's (`missFreeLut`).
            pub fn routeMissFree(self: *StreamSource, mf: *diag.MissFree, layer: u32, ids: []const u16) Error!*Call {
                var buf: [max_route_ids]u16 = undefined;
                const rel = self.relativeLayer(layer);
                const sub = mf.substitute(rel, ids, &buf);
                const r = try self.stream.route(rel, sub, &.{});
                std.debug.assert(r.plan.n_misses == 0 and r.plan.n_loads == 0 and r.n_parts == 0);
                if (comptime profile.enabled) if (self.trace) |t| t.decodeRoute(rel, sub, &r.plan, &.{});
                // The MTP profile's union: the rows' routed experts (before substitution), as the normal route counts it.
                const call = self.callOf(r);
                call.* = .{ .route = r, .n_ids = @intCast(ids.len) };
                _ = self.stream.refsOf(r, &call.refs);
                wavesOf(r, call.waves[0..ids.len]);
                return call;
            }

            /// The device LUT under missfree: every expert's entry is its substitute's packed slot (a resident's own).
            pub fn missFreeLut(self: *const StreamSource, mf: *const diag.MissFree, layer: u32, out: []u32) void {
                const rel = self.relativeLayer(layer);
                const pol = &self.stream.layers[rel].policy;
                for (out, 0..) |*o, e| {
                    o.* = 0;
                    if (e >= pol.n_experts) continue;
                    const ref = self.stream.slotRef(rel, pol.slotOf(mf.sub(rel, @intCast(e))).?);
                    o.* = (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
                }
            }
            // ── end MIMO_DIAG ──

            pub fn shrink(self: *StreamSource, rows: []const u32, residents: S.Residents) !u64 {
                return self.stream.shrink(rows, residents);
            }

            pub fn resumePrefill(self: *StreamSource, rows: []const u32) !void {
                // Same request and uniform retained rows: preserve its residents,
                // seed and prompt counts while restoring the prompt's window depth.
                const st = self.stream;
                if (std.Thread.getCurrentId() != st.owner) return error.NotInferenceThread;
                if (st.phase != .decode) return error.NotGrown;
                if (st.failed) return error.StreamFailed;
                if (st.release_installed or st.transient_released) return error.ResumeNeedsUniformRetainedRows;
                if (rows.len != st.layers.len) return error.InvalidRows;
                for (st.layers, rows) |*ls, n| {
                    if (n != ls.base.rows or n != ls.policy.capacity or ls.ext != null) return error.InvalidRows;
                }
                try st.settleRoutes();
                // Settled demand and synchronized consumers cannot still read slots.
                // Cancel/join the separate speculative horizon before prompt routing.
                try st.pool.quiesce(10 * std.time.ns_per_s);
                st.phase = .prefill;
                st.route_lookahead = false;
                st.route_preread = false;
                st.wide_depth = st.prompt_wide_depth;
            }

            pub fn releaseTransient(self: *StreamSource) !u64 {
                return self.stream.releaseTransient();
            }

            pub fn regrowTransient(self: *StreamSource) !u64 {
                return self.stream.regrowTransient();
            }

            pub fn forgetResidents(self: *StreamSource) !u32 {
                return self.stream.forgetResidents();
            }

            pub fn resetPromptCounts(self: *StreamSource) void {
                self.stream.resetPromptCounts();
            }

            pub fn readAheadSeed(self: *StreamSource, layer: u32, ids: []const u16) !void {
                return self.stream.readAheadSeed(self.relativeLayer(layer), ids);
            }

            pub fn awaitReadAhead(self: *StreamSource, layer: u32) Error!void {
                // X12 (`gpu_keep_warm: "waits"` / `"all"`): the host waits on the prompt's read-ahead bytes; warm meanwhile.
                // P7: through the installed read schedule's await.
                if (self.warm_arm) |w| {
                    w.holdOn(true);
                    defer w.holdOn(false);
                    return self.prompt_await(self, self.relativeLayer(layer));
                }
                return self.prompt_await(self, self.relativeLayer(layer));
            }

            pub fn checkReadAhead(self: *StreamSource, layer: u32, n: u32) !u32 {
                const p = self.policy(layer);
                var ids: [max_route_ids]u16 = undefined;
                var k: u32 = 0;
                var expert: u32 = p.n_experts;
                while (expert > 0 and k < @min(n, max_route_ids)) {
                    expert -= 1;
                    if (p.slotOf(@intCast(expert)) != null) continue;
                    ids[k] = @intCast(expert);
                    k += 1;
                }
                try self.stream.checkReadAhead(self.relativeLayer(layer), ids[0..k]);
                return k;
            }

            pub fn isResident(self: *const StreamSource, layer: u32, expert: u16) bool {
                return self.stream.layers[self.relativeLayer(layer)].policy.slotOf(expert) != null;
            }

            pub fn readWallNs(self: *StreamSource) u64 {
                return @intCast(@max(self.stream.pool.readGauge()[4], 0));
            }

            pub fn gate(self: *StreamSource, call: *Call) Error!?Gates {
                const gates = try self.stream.gate(call.route.?);
                // Profile builds: the prompt timeline's next tiles wait for the group's last down value (0: nothing read).
                if (comptime prefill_timeline.enabled) prefill_timeline.gate(if (gates) |gt| gt.down_first + gt.n_parts - 1 else 0);
                // X12 (`gpu_keep_warm: "waits"` / `"all"`): the call's gated parts (decode miss parts, a prompt group's reads)
                // wait for values up to the last down part's; the warm engine keeps the GPU busy until the stream's event
                // reaches it.
                if (self.warm_arm) |w| if (gates) |gt| w.call(gt.down_first + gt.n_parts - 1);
                return gates;
            }

            pub fn stats(self: *StreamSource) Stats {
                return self.stream.stats();
            }

            /// ExpertsWith binds all absolute model layers once. Layer 0 is dense.
            pub fn bankRows(self: *StreamSource, layer: u32, kind: BankKind) u32 {
                if (layer < self.stream.bank.first_moe_layer) return 0;
                const ls = &self.stream.layers[self.relativeLayer(layer)];
                return switch (kind) {
                    .base => ls.base.rows,
                    .ext => if (ls.ext) |e| e.rows else 0,
                    .transient => if (self.stream.transient_released) 0 else self.stream.transient.rows,
                };
            }

            pub fn bankArrays(self: *StreamSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
                const rows = self.bankRows(layer, kind);
                if (rows == 0) return null;
                const index = self.relativeLayer(layer);
                if (@TypeOf(g.*).T == mlx.mlx_array) {
                    return self.stream.bankArrays(index, kind);
                } else {
                    if (comptime !@hasDecl(@TypeOf(g.*), "input")) @compileError("MXFP4 StreamSource requires MLX arrays or trace inputs");
                    return try traceBank(g, &self.stream.bank.layers[index], rows);
                }
            }
        };
    };
}
