//! GLM-5.3's routed experts over the expert stream. Per routed-layer call the trunk hands the driver the MoE input
//! `x [n, hidden]`, the router's top-k `indices [n, k]` (int32) and their weights `scores [n, k]` (f32, normalized and
//! scaled), and gets back the weighted sum `[n, hidden]` in x's dtype (the reference's `(y * scores[..., None]).sum(-2)`).
//! The bank module `Bk` (its stream, its slot arrays) and the routed-expert math `M` (the C2 quant contract:
//! `gateUp`, `down`, `prefill`, `finishPrefill`) are comptime parameters: the affine bank binds MLX's `gather_qmm`
//! (`sdk_ext.quant.FromGatherMatmul(GatherQmm)`), another bank binds its own quant with no change here or in the trunk.
//!
//! A call of at most `max_route_ids` routed ids (decode, short prompts) is the decode lane: the routing barrier (the ids
//! and, with the lookahead, the next routed layer's scores read on the host), the route, the residents' wave at once,
//! then each miss part's gate/up after its gate/up bytes landed and its down after the rest (or every wave at once over
//! event-gate aliases), the release; the outputs joined in routed order. A wider call is the wide lane, per bank layer:
//! its residency seeded from the call's ids, the call's experts in groups of `max_route_ids` (its residents first, then
//! the misses read at the layer's start (`stageMisses`), then the rest, each hottest first) routed up to `wide_depth`
//! ahead, each group's rows per bank through the math's `prefill` in slices, committed as built and drained (while the
//! host builds the next) before the group's slots go back; then the combine (on the GPU one launch over the groups' outputs, else the join and the op chain in token slices).

const std = @import("std");
const env = @import("mlx_host").env;
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const expert_policy = sdk_ext.expert.policy;
const expert_event = sdk_ext.expert.event;
const expert_stream = @import("../../sdk/weights/streamed/expert/stream.zig");
const ops = @import("../../sdk/weights/streamed/ops.zig");
const trunk = @import("trunk_kernels.zig");
const glm = @import("glm_moe_dsa.zig");
/// PROFILE builds (`-Dplugin-profile=true`): each decode-lane call's host stamps on the verify's GPU timeline.
const timeline = @import("../../sdk/weights/streamed/profile/verify_timeline.zig");
const pt = @import("prefill_timers.zig");

/// The parts one combine launch reads (`trunk_kernels.max_combine_parts`: Metal binds at most 31 buffers).
const max_combine_parts = trunk.max_combine_parts;

/// `(y * scores[..., None]).sum(-2)` of the routed rows in `parts` (at most `max_combine_parts`; token t's j-th routed
/// row is row `loc[t * k + j] & 0xFFFFFF` of part `loc[t * k + j] >> 24`, u32 `[n * k]`) as one launch of the trunk
/// registry's combine (`glm_moe_dsa_combine`: one thread per token and 4 hidden values, the token's `k` routed rows in
/// routed order, weighted and summed in f32, products and sums rounded apart, cast to `dt`): the op chain's bits.
fn combineOnGpu(bound: *const trunk.Bound, g: *ops.MlxOps, parts: []const ops.MlxOps.T, loc: ops.MlxOps.T, scores: ops.MlxOps.T, n: u32, k: u32, hidden: c_int, dt: ops.Dtype) !ops.MlxOps.T {
    std.debug.assert(parts.len > 0 and parts.len <= max_combine_parts and @rem(hidden, 4) == 0);
    // The slots past the parts hold the first part (never read).
    var inputs: [trunk.max_inputs]mlx.mlx_array = undefined;
    for (inputs[0..max_combine_parts], 0..) |*in, i| in.* = if (i < parts.len) parts[i] else parts[0];
    inputs[max_combine_parts] = loc;
    inputs[max_combine_parts + 1] = scores;
    const cfg = trunk.combineLaunch(n, k, @intCast(hidden), dt);
    var outs: [1]mlx.mlx_array = undefined;
    try bound.apply(.glm_moe_dsa_combine, &inputs, &cfg, &outs);
    return g.adopt(outs[0]);
}

pub const BankKind = sdk_ext.expert.BankKind;
pub const SlotRef = sdk_ext.expert.SlotRef;
pub const max_route_ids = expert_policy.max_route_ids;
const n_banks = std.meta.fieldNames(BankKind).len;

/// The driver's construction-time routes.
/// The decode lane's host time per call, summed: blocked in the routing barrier (the GPU up to the router, with event
/// gates also its wait on the previous layer's bytes), the routes (the ids read, the plans, the reads submitted, the
/// lookahead's step), the waves' encode and commit.
/// `joined`: the calls whose outputs were more parts than the GPU combine reads (`max_combine_parts`), joined first.
pub const HostSplit = struct { calls: u64 = 0, barrier_ns: u64 = 0, route_ns: u64 = 0, wave_ns: u64 = 0, joined: u64 = 0 };

pub const Options = struct {
    /// The waves wait on the reads' event gates (the stream built with `event`); off: the host waits.
    gated: bool = false,
    /// The routed combine on the trunk set's kernel (`combineOnGpu`): the construction's route on a GPU stream at an
    /// instantiation the registry serves (the module's trunk set, its compile check run there); null: the op chain's
    /// join and combine.
    combine: ?*const trunk.Bound = null,
    /// The MLX event the stream signals (gated MLX backends).
    event: ?expert_event.Event = null,
    /// The wide lane's groups in flight (at most the stream's `wide_depth`).
    wide_depth: u8 = 1,
};

pub fn Experts(comptime G: type, comptime Bk: type, comptime M: type) type {
    comptime sdk_ext.expert.assertBank(Bk);
    return struct {
        const Self = @This();
        const T = G.T;
        const S = Bk.Stream;
        pub const Stream = S.Stream;
        pub const Arrays = Bk.BankArrays;
        /// Stream layers per routed layer (GLM-5.3's EXL3 bank: its K3 and K4 bank layers): a call's ids split by the
        /// bank layer that holds each expert, one route each.
        pub const bpl: u32 = if (@hasDecl(Bk, "banks_per_layer")) Bk.banks_per_layer else 1;
        /// The quant fuses gate, up and down (`M.fused`): each wave runs once all its segments landed.
        const fused = @hasDecl(M, "fused_waves") and M.fused_waves;
        comptime {
            if (bpl > 1 and !fused) @compileError("a bank of several bank layers per routed layer streams through a fused quant");
        }

        a: std.mem.Allocator,
        stream: *Stream,
        math: *M,
        hidden: c_int,
        n_experts: u32,
        /// Per routed layer, per bank kind: what the math binds (null: no rows, or released).
        banks: [][n_banks]?Arrays,
        opt: Options,
        transient_released: bool = false,
        wide: Wide = .{},
        staged: Staged = .{},
        /// The decode lane's read bytes by the first row of the call that routes the expert (row i of `x`), summed over
        /// calls until the caller zeroes it: a draft round's verify attributes its reads to the rows it keeps or rejects.
        row_bytes: [max_route_ids]u64 = @splat(0),
        /// The decode lane's host time, summed over calls until the caller zeroes it (`HostSplit`).
        host: HostSplit = .{},
        /// The combine on the GPU (`Options.combine`, the trunk set's kernel); null: the op chain's combine.
        combine_kernel: ?*const trunk.Bound = null,
        /// Test builds only: each decode-lane call's GPU combine also built as the op chain's join and combine and
        /// compared bit for bit (`checked` the calls compared, `differed` those whose outputs differ). A served build has
        /// no field and no branch for it.
        combine_check: if (builtin.is_test) CombineCheck else void = if (builtin.is_test) .{} else {},

        const CombineCheck = struct { on: bool = false, checked: u32 = 0, differed: u32 = 0 };

        /// The routes of the prompt layer whose misses were read at its start (`stageMisses`): each route's bank layer and
        /// experts, in route order; null once the call took it.
        const Staged = struct {
            layer: ?u32 = null,
            routes: [expert_stream.max_wide_depth]?*S.Route = @splat(null),
            sls: [expert_stream.max_wide_depth]u32 = undefined,
            n_routes: usize = 0,
            experts: std.ArrayList(u16) = .empty,
            bounds: [expert_stream.max_wide_depth + 1]usize = undefined,
        };

        /// The wide lane's host scratch, reused across calls.
        const Wide = struct {
            ids: std.ArrayList(u16) = .empty,
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            order: std.ArrayList(u16) = .empty,
            count: std.ArrayList(u32) = .empty,
            /// The call's routed rows by group (each group's in row order) and each group's first index there.
            by_group: std.ArrayList(u32) = .empty,
            group_at: std.ArrayList(u32) = .empty,
            /// The groups: their bounds in `order`, each expert's group, each group's route.
            bounds: std.ArrayList(u32) = .empty,
            egroup: std.ArrayList(u16) = .empty,
            groute: std.ArrayList(?*S.Route) = .empty,
            staged_mark: std.ArrayList(bool) = .empty,
            slot: std.ArrayList(u32) = .empty,
            act: std.ArrayList(u32) = .empty,
            pos: std.ArrayList(u32) = .empty,
            inv: std.ArrayList(u32) = .empty,
            kept: std.ArrayList(T) = .empty,
            side_ids: std.ArrayList(u16) = .empty,
            /// The call's GEMM slices in flight (indices into `kept`): the slice window (`glm.prompt_gemm_slices_in_flight`).
            window: SliceWindow = .{},

            fn deinit(w: *Wide, a: std.mem.Allocator, g: *G) void {
                for (w.kept.items) |x| g.release(x);
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.order, &w.count, &w.by_group, &w.group_at, &w.bounds, &w.egroup, &w.groute, &w.staged_mark, &w.slot, &w.act, &w.pos, &w.inv, &w.kept, &w.side_ids }) |l| l.deinit(a);
            }
        };

        /// A routed call's GEMM slices in flight, at most `glm.prompt_gemm_slices_in_flight`: each slice is committed as
        /// it is built (`commit`), and before the next is built the oldest still in flight is waited for (`admit`), so
        /// its transients are back in MLX's cache. The bill charges exactly this (`bill.gemm_rows_in_flight_window`): the
        /// schedule bounds the slices' transients on any box, not the GPU's pace against the host's.
        const SliceWindow = struct {
            const cap: usize = glm.prompt_gemm_slices_in_flight;
            ring: [cap]usize = undefined,
            head: usize = 0,
            n: usize = 0,

            fn admit(sw: *SliceWindow, g: *G, kept: []const T) !void {
                if (sw.n < cap) return;
                try g.evalAll(&.{kept[sw.ring[sw.head]]});
                sw.head = (sw.head + 1) % cap;
                sw.n -= 1;
            }

            fn commit(sw: *SliceWindow, g: *G, kept: []const T, at: usize) !void {
                try g.asyncEval(&.{kept[at]});
                sw.ring[(sw.head + sw.n) % cap] = at;
                sw.n += 1;
            }
        };

        pub fn init(a: std.mem.Allocator, g: *G, stream: *Stream, math: *M, hidden: u32, opt: Options) !Self {
            if (opt.gated and !stream.gated) return error.GatedNeedsStreamEvent;
            if (opt.gated and G == ops.MlxOps and opt.event == null) return error.GatedNeedsEvent;
            if (opt.wide_depth < 1 or opt.wide_depth > stream.wide_depth) return error.WideDepthExceedsStream;
            const banks = try a.alloc([n_banks]?Arrays, stream.layers.len);
            errdefer a.free(banks);
            for (banks, 0..) |*b, l| {
                b.* = @splat(null);
                for ([_]BankKind{ .base, .transient }) |kind| b[@backingInt(kind)] = try bind(stream, @intCast(l), kind);
            }
            // The combine route: the trunk set's kernel the module built and checked at its construction, never in a request.
            const combine_kernel: ?*const trunk.Bound = if (comptime G == ops.MlxOps) opt.combine else null;
            if (combine_kernel != null) std.debug.assert(mlx.streamIsGpu(g.s) and hidden % 4 == 0);
            return .{ .a = a, .stream = stream, .math = math, .hidden = @intCast(hidden), .n_experts = stream.bank.n_experts, .banks = banks, .opt = opt, .combine_kernel = combine_kernel };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.staged.experts.deinit(self.a);
            self.wide.deinit(self.a, g);
            self.a.free(self.banks);
            self.* = undefined;
        }

        /// A bank with rows must have arrays the math binds (MLX slot memory).
        fn bind(stream: *Stream, layer: u32, kind: BankKind) !?Arrays {
            const arrays = stream.bankArrays(layer, kind);
            if (arrays == null and rowsOf(stream, layer, kind) > 0) return error.SlotArraysUnbound;
            return arrays;
        }

        fn rowsOf(stream: *Stream, layer: u32, kind: BankKind) u32 {
            const ls = &stream.layers[layer];
            return switch (kind) {
                .base => ls.base.rows,
                .ext => if (ls.ext) |e| e.rows else 0,
                .transient => stream.transient.rows,
            };
        }

        /// The phase change's first free: every transient binding nulled, then the stream frees the scratch.
        pub fn releaseTransient(self: *Self) !u64 {
            for (self.banks) |*b| b[@backingInt(BankKind.transient)] = null;
            self.transient_released = true;
            return self.stream.releaseTransient();
        }

        /// The one phase change: the stream grows; the grown rows and decode's window 0 are bound.
        pub fn grow(self: *Self, decode_rows: []const u32) !void {
            try self.stream.grow(decode_rows);
            for (self.banks, 0..) |*b, l| {
                b[@backingInt(BankKind.ext)] = try bind(self.stream, @intCast(l), .ext);
                b[@backingInt(BankKind.transient)] = try bind(self.stream, @intCast(l), .transient);
            }
            self.transient_released = false;
        }

        /// The reverse phase change's free: the grown rows' and window 0's bindings nulled, then the stream frees them.
        pub fn shrink(self: *Self, prompt_rows: []const u32, residents: S.Residents) !u64 {
            for (self.banks) |*b| {
                b[@backingInt(BankKind.ext)] = null;
                b[@backingInt(BankKind.transient)] = null;
            }
            self.transient_released = true;
            return self.stream.shrink(prompt_rows, residents);
        }

        /// The reverse phase change's allocation: the prompt's scratch re-created and bound.
        pub fn regrowTransient(self: *Self) !u64 {
            const bytes = try self.stream.regrowTransient();
            for (self.banks, 0..) |*b, l| b[@backingInt(BankKind.transient)] = try bind(self.stream, @intCast(l), .transient);
            self.transient_released = false;
            return bytes;
        }

        /// After the forward's last eval: settles and unpins released calls.
        pub fn flush(self: *Self) !void {
            // The call's promotions landed before anything reads their rows (the next request, decode's grow).
            self.stream.settleCopies();
            try self.stream.flush();
        }

        /// The stream is in its prompt phase (before `grow`): a prompt call's predictor has rows to read ahead into. A
        /// multi-row step past the phase change (a decode-phase `extend`) predicts nothing.
        pub fn prompting(self: *const Self) bool {
            return self.stream.phase == .prefill;
        }

        /// P1: layer `layer`'s predicted experts (hottest first) read ahead of its routed call (prompt phase).
        pub fn readAheadSeed(self: *Self, layer: u32, experts: []const u16) !void {
            if (bpl == 1) return self.stream.readAheadSeed(layer, experts) else {
                // One read-ahead is live at a time: the first bank layer's share of the seed, hottest first.
                var buf: [512]u16 = undefined;
                var n: usize = 0;
                for (experts) |e| if (n < buf.len and self.stream.bank.streamLayer(layer, e) == bpl * layer) {
                    buf[n] = e;
                    n += 1;
                };
                return self.stream.readAheadSeed(bpl * layer, buf[0..n]);
            }
        }

        /// A wide prompt call's misses read at its layer's start, before its attention: each bank layer's non-resident
        /// experts (in expert order) routed into transient rows in groups, as many as all windows but two hold (the
        /// call's groups alternate in those two; a bank layer's residents are held meanwhile: no staged miss evicts one);
        /// `callWide` serves them from these routes. A call of at least 8 rows per expert routes nearly every expert, so
        /// nearly every staged read is used.
        pub fn stageMisses(self: *Self, layer: u32) !void {
            const st = &self.staged;
            std.debug.assert(st.layer == null);
            if (self.opt.wide_depth < 3) return;
            const max_routes: usize = self.opt.wide_depth - 2;
            const group_n = self.stream.max_route_ids;
            st.experts.clearRetainingCapacity();
            st.bounds[0] = 0;
            st.n_routes = 0;
            st.layer = layer;
            errdefer self.dropStaged();
            for (0..bpl) |side| {
                if (st.n_routes == max_routes) break;
                const sl = bpl * layer + @as(u32, @intCast(side));
                const policy = &self.stream.layers[sl].policy;
                const e0 = st.experts.items.len;
                for (0..self.n_experts) |e| {
                    const id: u16 = @intCast(e);
                    if (bpl > 1 and self.stream.bank.streamLayer(layer, id) != sl) continue;
                    if (policy.slotOf(id) == null) try st.experts.append(self.a, id);
                }
                if (st.experts.items.len == e0) continue;
                try self.stream.holdResidents(sl);
                defer self.stream.releaseHeld();
                var lo = e0;
                while (lo < st.experts.items.len and st.n_routes < max_routes) {
                    const hi = @min(lo + group_n, st.experts.items.len);
                    st.routes[st.n_routes] = try self.stream.route(sl, st.experts.items[lo..hi], &.{});
                    st.sls[st.n_routes] = sl;
                    st.n_routes += 1;
                    st.bounds[st.n_routes] = hi;
                    lo = hi;
                }
                st.experts.shrinkRetainingCapacity(lo);
            }
            if (st.n_routes == 0) st.layer = null;
        }

        /// Group `gi`'s waves (`outs`, committed) drained before its slots go back (the next route may refill them), its
        /// route released, the groups after it routed on stream layer `sl` while windows are free.
        fn drainGroup(self: *Self, g: *G, w: *Wide, sl: u32, gi: usize, outs: []const T, next: *usize, live: *usize, depth: usize, comptime ahead: anytype) !void {
            const td = pt.now();
            try g.evalAll(outs);
            pt.chargeRouted(.compute, td);
            const tr = pt.now();
            const r = w.groute.items[gi].?;
            var cand: Seeds = .{};
            seedsOf(&self.stream.layers[sl].policy, w.order.items[w.bounds.items[gi]..w.bounds.items[gi + 1]], r.plan.slotsOf(), &cand);
            self.stream.release(r);
            w.groute.items[gi] = null;
            live.* -= 1;
            if (cand.n > 0) {
                const tp = pt.now();
                // The eval above consumed the group: its route unpinned (its rows the policy's again), then its seed
                // experts' copies handed to the copier (a route into their window waits for them).
                try self.stream.flush();
                var items: [max_route_ids]S.Stream.Promotion = undefined;
                for (cand.items[0..cand.n], items[0..cand.n]) |cd, *it| it.* = .{ .e = cd.e, .from = cd.slot };
                _ = self.stream.promote(sl, items[0..cand.n]);
                pt.chargeRouted(.promote, tp);
            }
            try ahead(self, w, sl, next, live, depth);
            pt.chargeRouted(.route, tr);
        }

        /// A group's seed experts (the call's hottest, `LayerPolicy.seed`) that its route read into transient rows (a
        /// staged route, planned before the seed was known): each expert and its transient slot, hottest first.
        const Seeds = struct { items: [max_route_ids]struct { e: u16, slot: u32, n: u32 } = undefined, n: usize = 0 };

        fn seedsOf(pol: *const expert_policy.LayerPolicy, exps: []const u16, slots: []const u32, out: *Seeds) void {
            for (exps, slots) |e, slot| {
                if (slot < pol.capacity or !pol.seed.isSet(e)) continue;
                out.items[out.n] = .{ .e = e, .slot = slot, .n = pol.call_counts[e] };
                out.n += 1;
            }
            const C = @TypeOf(out.items[0]);
            std.sort.insertion(C, out.items[0..out.n], {}, struct {
                fn lt(_: void, x: C, y: C) bool {
                    return if (x.n != y.n) x.n > y.n else x.e < y.e;
                }
            }.lt);
        }

        /// The staged routes the call did not take released (its end, or an error between `stageMisses` and the call).
        pub fn dropStaged(self: *Self) void {
            const st = &self.staged;
            if (st.layer == null) return;
            for (st.routes[0..st.n_routes]) |*rt| if (rt.*) |r| {
                self.stream.release(r);
                rt.* = null;
            };
            st.layer = null;
            st.n_routes = 0;
        }

        /// One routed-layer call: x [n, hidden], indices [n, k] int32, scores [n, k] f32 -> the weighted sum [n, hidden]
        /// in x's dtype. `next_scores` [n, n_experts] f32: the next routed layer's routing scores on this call's rows (the
        /// decode lookahead's predictor), evaluated with the ids at the routing barrier; null for none. `hoist`: arrays
        /// that do not wait on this call (the shared expert), committed behind the hit wave once the reads are issued, so
        /// the GPU runs them during the read wait (the decode lane; the wide lane leaves them to the caller).
        pub fn call(self: *Self, g: *G, layer: u32, x: T, indices: T, scores: T, next_scores: ?T, hoist: []const T) !T {
            const n: u32 = @intCast(g.shapeOf(x).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            if (n * k > self.stream.max_route_ids) return self.callWide(g, layer, x, indices, scores, n, k);
            const acc = if (bpl > 1) try self.callDecodeSplit(g, layer, x, indices, next_scores, n, k, hoist) else try self.callDecode(g, layer, x, indices, next_scores, n, k, hoist);
            return self.joinCombine(g, &acc, scores, n, k, g.dtypeOf(x));
        }

        /// The decode lane's outputs `acc` (in wave order; `acc.pos` their routed positions) weighted by `scores` and
        /// summed per row, in `dt`: on the GPU, at most `max_combine_parts` outputs, the combine reads each routed row
        /// where its wave left it (`combineOnGpu`: the op chain's bits); else the rows joined in routed order and
        /// `combine`.
        fn joinCombine(self: *Self, g: *G, acc: *const Acc, scores: T, n: u32, k: u32, dt: ops.Dtype) !T {
            const n_ids = n * k;
            const outs = acc.outs[0..acc.n_outs];
            // The construction's route (`combine_kernel`), then the call's own value: the parts the kernel binds.
            if (comptime G == ops.MlxOps) if (self.combine_kernel) |kern| if (outs.len <= max_combine_parts) {
                var loc: [max_route_ids]u32 = undefined;
                var j: usize = 0;
                for (outs, 0..) |part, pi| for (0..@intCast(g.shapeOf(part).dim(0))) |row| {
                    loc[acc.pos[j]] = @as(u32, @intCast(pi)) << 24 | @as(u32, @intCast(row));
                    j += 1;
                };
                const lc = try g.hostArray(std.mem.sliceAsBytes(loc[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
                const out = try combineOnGpu(kern, g, outs, lc, scores, n, k, self.hidden, dt);
                if (comptime builtin.is_test) if (self.combine_check.on) {
                    // f32 holds every bf16 bit (both zeros apart): the bits compared on the host.
                    const want = try combine(g, try self.joinRows(g, acc, n, k), scores, dt);
                    const len = n * @as(usize, @intCast(self.hidden));
                    const got_v = try self.a.alloc(f32, 2 * len);
                    defer self.a.free(got_v);
                    _ = try g.hostF32(try g.astype(out, .float32), got_v[0..len]);
                    _ = try g.hostF32(try g.astype(want, .float32), got_v[len..]);
                    self.combine_check.checked += 1;
                    if (!std.mem.eql(u8, std.mem.sliceAsBytes(got_v[0..len]), std.mem.sliceAsBytes(got_v[len..]))) self.combine_check.differed += 1;
                };
                return out;
            };
            self.host.joined += 1;
            return combine(g, try self.joinRows(g, acc, n, k), scores, dt);
        }

        /// `acc`'s rows in routed order, `[n, k, hidden]`.
        fn joinRows(self: *const Self, g: *G, acc: *const Acc, n: u32, k: u32) !T {
            const n_ids = n * k;
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            for (acc.pos[0..acc.n_pos], 0..) |p, j| order[p] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// `(y * scores[..., None]).sum(-2)` in f32, cast to `dt` (the reference's combine).
        fn combine(g: *G, y: T, scores: T, dt: ops.Dtype) !T {
            const w = try g.expandDims(scores, -1);
            return g.astype(try g.sum(try g.mul(y, w), -2, false), dt);
        }

        const Group = struct { bank: BankKind, n: u32 = 0, pos: [max_route_ids]u8 = undefined };
        const Wave = struct {
            groups: [n_banks]Group = undefined,
            n: usize = 0,
            ids: [n_banks]T = undefined,
            h: [n_banks]T = undefined,
        };
        const Acc = struct {
            outs: [2 * max_route_ids]T = undefined,
            n_outs: usize = 0,
            pos: [max_route_ids]u32 = undefined,
            n_pos: usize = 0,
        };

        /// Wave `w`'s positions grouped by bank in first appearance, each group's gate/up and activation.
        fn gateUpWave(self: *Self, g: *G, layer: u32, x: T, k: u32, sv: sdk_ext.expert.Served, w: u8, over: ?*const [n_banks]?Arrays) !Wave {
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
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]) orelse return error.SlotArraysUnbound;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |pos, *t, *r| {
                    t.* = @intCast(pos / k);
                    r.* = sv.refs[pos].row;
                }
                const nn: c_int = @intCast(gr.n);
                const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{nn}, .int32), 0);
                wave.ids[i] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{nn}, .uint32);
                wave.h[i] = try self.math.gateUp(g, xs, wave.ids[i], arrays.gate, arrays.up);
            }
            return wave;
        }

        fn downWave(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) ![]const T {
            const first = acc.n_outs;
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]) orelse return error.SlotArraysUnbound;
                acc.outs[acc.n_outs] = try self.math.down(g, wave.h[i], wave.ids[i], arrays.down);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |pos| {
                    acc.pos[acc.n_pos] = pos;
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

        /// The event-wait aliases of every array of the projections in `ps` (a struct of projections, or of optional
        /// ones), read by the GPU after `value` and `deps`: one wait per array (`waitArrays`).
        fn waitAll(self: *Self, g: *G, ps: anytype, value: u64, deps: []const T) !@TypeOf(ps) {
            const P = @TypeOf(ps);
            var out = ps;
            inline for (comptime std.meta.fieldNames(P)) |pf| {
                const p = @field(ps, pf);
                if (@typeInfo(@TypeOf(p)) == .optional) {
                    if (p) |q| @field(out, pf) = try self.waitArrays(g, q, value, deps);
                } else @field(out, pf) = try self.waitArrays(g, p, value, deps);
            }
            return out;
        }

        /// An event wait's aliases of every array of one projection (`outs` fresh), read by the GPU after `value`.
        fn waitArrays(self: *Self, g: *G, p: anytype, value: u64, deps: []const T) !@TypeOf(p) {
            var out = p;
            inline for (comptime std.meta.fieldNames(@TypeOf(p))) |f| {
                const v = @field(p, f);
                const x: ?T = if (@TypeOf(v) == T) v else v;
                if (x) |arr| {
                    var o = [1]T{undefined};
                    if (G == ops.MlxOps) {
                        o[0] = mlx.mlx_array_new();
                        try expert_event.wait(&.{arr}, self.opt.event.?, value, deps, false, g.s, &o);
                        o[0] = try g.adopt(o[0]);
                    } else o[0] = try g.eventAlias(arr, value, deps.len);
                    @field(out, f) = o[0];
                }
            }
            return out;
        }

        /// The gated waves: every part's gate/up over bank arrays waited at `gu`, then part p's down over arrays
        /// waited at `down_first + p`, after every gate/up and the previous part's down. Nothing waits on the host.
        fn gatedParts(self: *Self, g: *G, layer: u32, x: T, k: u32, sv: sdk_ext.expert.Served, gates: sdk_ext.expert.Gates, acc: *Acc) !void {
            var gu: [n_banks]?Arrays = @splat(null);
            for (sv.waves, sv.refs) |w, ref| {
                const b = @backingInt(ref.bank);
                if (w == 0 or gu[b] != null) continue;
                const arrays = self.banks[layer][b] orelse return error.SlotArraysUnbound;
                const w2 = try self.waitAll(g, .{ .gate = arrays.gate, .up = arrays.up }, gates.gu, &.{});
                gu[b] = .{ .gate = w2.gate, .up = w2.up, .down = arrays.down };
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gateUpWave(g, layer, x, k, sv, @intCast(p + 1), &gu);
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
                const d0 = if (dn[0]) |arr| arr.down else null;
                const d1 = if (dn[1]) |arr| arr.down else null;
                const d2 = if (dn[2]) |arr| arr.down else null;
                const wd = try self.waitAll(g, .{ .b0 = d0, .b1 = d1, .b2 = d2 }, gates.down_first + p, deps[0 .. n_hs + prev.len]);
                if (dn[0]) |*arr| arr.down = wd.b0.?;
                if (dn[1]) |*arr| arr.down = wd.b1.?;
                if (dn[2]) |*arr| arr.down = wd.b2.?;
                prev = try self.downWave(g, layer, &waves[p], acc, &dn);
            }
        }

        /// The decode lane: the routed outputs in wave order and their routed positions (`Acc`).
        fn callDecode(self: *Self, g: *G, layer: u32, x: T, indices: T, next_scores: ?T, n: u32, k: u32, hoist: []const T) !Acc {
            const n_ids = n * k;
            const t0 = timeline.now();
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [sdk_ext.expert.lookahead.max_rows * 512]f32 = undefined;
            var sc: []const f32 = &.{};
            const read_next = next_scores != null and self.stream.route_lookahead and n * self.n_experts <= score_buf.len;
            if (read_next) {
                try g.evalAll(&.{ indices, next_scores.? });
                sc = try g.hostF32(next_scores.?, score_buf[0 .. n * self.n_experts]);
            }
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const t1 = timeline.now();
            const r = try self.stream.route(layer, ids, sc);
            const t2 = timeline.now();
            defer self.noteHost(t0, t1, t2);
            var released = false;
            errdefer if (!released) self.stream.release(r);
            const record = self.stream.bank.layers[layer].logical_bytes;
            for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, read| if (read) {
                const at = std.mem.indexOfScalar(u16, ids, l.expert) orelse continue;
                self.row_bytes[at / k] += record;
            };
            var refs: [max_route_ids]SlotRef = undefined;
            var waves: [max_route_ids]u8 = undefined;
            _ = self.stream.refsOf(r, &refs);
            var bufs: [max_route_ids][max_route_ids]expert_policy.Load = undefined;
            var parts: [max_route_ids][]const expert_policy.Load = undefined;
            for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
            expert_stream.wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], waves[0..n_ids]);
            const sv: sdk_ext.expert.Served = .{ .refs = refs[0..n_ids], .waves = waves[0..n_ids], .n_parts = r.n_parts };
            var acc: Acc = .{};
            const hits = try self.gateUpWave(g, layer, x, k, sv, 0, null);
            // The hit wave and the hoisted arrays in one commit, ahead of every wait on a read.
            var early: [n_banks + 4]T = undefined;
            var n_early: usize = 0;
            if (hits.n > 0) for (try self.downWave(g, layer, &hits, &acc, null)) |o| {
                early[n_early] = o;
                n_early += 1;
            };
            for (hoist) |h| {
                if (n_early == early.len) break;
                early[n_early] = h;
                n_early += 1;
            }
            if (n_early > 0) try g.asyncEval(early[0..n_early]);
            if (self.opt.gated) {
                if (try self.stream.gate(r)) |gates| try self.gatedParts(g, layer, x, k, sv, gates, &acc);
            } else for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                try self.stream.waitGu(r, part);
                const wave = try self.gateUpWave(g, layer, x, k, sv, @intCast(p + 1), null);
                try g.asyncEval(wave.h[0..wave.n]);
                try self.stream.waitDown(r, part);
                try g.asyncEval(try self.downWave(g, layer, &wave, &acc, null));
            }
            self.stream.release(r);
            released = true;
            return acc;
        }

        /// One route of a split call: its stream layer, its ids and their positions in the call (routed order), its
        /// slot refs and each position's wave.
        const Side = struct {
            sl: u32 = 0,
            n: u32 = 0,
            ids: [max_route_ids]u16 = undefined,
            pos: [max_route_ids]u8 = undefined,
            r: ?*S.Route = null,
            refs: [max_route_ids]SlotRef = undefined,
            waves: [max_route_ids]u8 = undefined,
        };

        /// The decode lane over a routed layer's `bpl` bank layers: the ids split by the bank layer that holds each
        /// expert and every route made before any wave (their reads in flight together; the last route settles the
        /// lookahead and reads ahead the next routed layer's experts over all its bank layers), every route's hit wave
        /// and the hoisted arrays in one commit, then each route's parts as fused waves, each once all its segments
        /// landed; the outputs in wave order and their routed positions (`Acc`).
        fn callDecodeSplit(self: *Self, g: *G, layer: u32, x: T, indices: T, next_scores: ?T, n: u32, k: u32, hoist: []const T) !Acc {
            const n_ids = n * k;
            const bank = self.stream.bank;
            const t0 = timeline.now();
            if (comptime timeline.enabled) timeline.point(layer, .call, self.readGauge(), false);
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [sdk_ext.expert.lookahead.max_rows * 512]f32 = undefined;
            var sc: []const f32 = &.{};
            const read_next = next_scores != null and self.stream.route_lookahead and n * self.n_experts <= score_buf.len;
            if (read_next) {
                try g.evalAll(&.{ indices, next_scores.? });
                sc = try g.hostF32(next_scores.?, score_buf[0 .. n * self.n_experts]);
            }
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const t1 = timeline.now();
            if (comptime timeline.enabled) timeline.point(layer, .barrier, 0, false);
            var sides: [bpl]Side = @splat(.{});
            for (&sides, 0..) |*sd, i| sd.sl = bpl * layer + @as(u32, @intCast(i));
            for (ids, 0..) |e, p| {
                const sd = &sides[bank.streamLayer(layer, e) - bpl * layer];
                sd.ids[sd.n] = e;
                sd.pos[sd.n] = @intCast(p);
                sd.n += 1;
            }
            errdefer for (&sides) |*sd| if (sd.r) |r| {
                self.stream.release(r);
                sd.r = null;
            };
            var last: usize = 0;
            for (sides, 0..) |sd, i| if (sd.n > 0) {
                last = i;
            };
            // The route recorder (a diagnostic): one route of the call, each bank layer's route flags its own ids.
            if (comptime sdk_ext.expert.stream.recording) if (self.stream.recorder) |rec| if (self.stream.phase == .decode) rec.call(layer, ids);
            defer if (comptime sdk_ext.expert.stream.recording) if (self.stream.recorder) |rec| rec.endCall();
            for (&sides, 0..) |*sd, i| {
                if (sd.n == 0) continue;
                const r = if (i == last) try self.stream.route(sd.sl, sd.ids[0..sd.n], sc) else try self.stream.routeHeld(sd.sl, sd.ids[0..sd.n]);
                sd.r = r;
                const geom = &bank.layers[sd.sl];
                const record = @as(u64, S.minisOf(geom)) * geom.logical_bytes;
                for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, read| if (read) {
                    const at = std.mem.indexOfScalar(u16, sd.ids[0..sd.n], l.expert) orelse continue;
                    self.row_bytes[sd.pos[at] / k] += record;
                };
                _ = self.stream.refsOf(r, &sd.refs);
                var bufs: [max_route_ids][max_route_ids]expert_policy.Load = undefined;
                var parts: [max_route_ids][]const expert_policy.Load = undefined;
                for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
                expert_stream.wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], sd.waves[0..sd.n]);
            }
            if (comptime timeline.enabled) timeline.point(layer, .route, 0, false);
            const t2 = timeline.now();
            defer self.noteHost(t0, t1, t2);
            var acc: Acc = .{};
            // Every route's hit wave and the hoisted arrays in one commit, ahead of every wait on a read.
            var early: [bpl * n_banks + 4]T = undefined;
            var n_early: usize = 0;
            for (&sides) |*sd| if (sd.r != null) for (try self.fusedWave(g, sd, x, k, 0, &acc, null)) |o| {
                early[n_early] = o;
                n_early += 1;
            };
            for (hoist) |h| {
                if (n_early == early.len) break;
                early[n_early] = h;
                n_early += 1;
            }
            if (n_early > 0) try g.asyncEval(early[0..n_early]);
            if (comptime timeline.enabled) timeline.point(layer, .hit, 0, false);
            var prev: []const T = &.{};
            var missed = false;
            var gu_first: u64 = 0;
            for (&sides) |*sd| if (sd.r) |r| {
                missed = missed or r.n_parts > 0;
                if (self.opt.gated) {
                    const gates = (try self.stream.gate(r)) orelse continue;
                    if (gu_first == 0) gu_first = gates.gu;
                    // The layer's gate/up value is its first route's, its last down value its last route's.
                    if (comptime timeline.enabled) timeline.gate(layer, gu_first, gates.down_first, gates.n_parts);
                    for (0..r.n_parts) |p| {
                        // Part p's banks waited at its down gate: every gate/up of the route and its own down landed.
                        var over: [n_banks]?Arrays = @splat(null);
                        for (sd.waves[0..sd.n], sd.refs[0..sd.n]) |w, ref| {
                            const b = @backingInt(ref.bank);
                            if (w != p + 1 or over[b] != null) continue;
                            const arrays = self.banks[sd.sl][b] orelse return error.SlotArraysUnbound;
                            over[b] = try self.waitAll(g, arrays, gates.down_first + p, prev);
                        }
                        prev = try self.fusedWave(g, sd, x, k, @intCast(p + 1), &acc, &over);
                    }
                } else for (0..r.n_parts) |p| {
                    try self.stream.waitGu(r, @intCast(p));
                    try self.stream.waitDown(r, @intCast(p));
                    try g.asyncEval(try self.fusedWave(g, sd, x, k, @intCast(p + 1), &acc, null));
                }
            };
            if (comptime timeline.enabled) timeline.point(layer, .end, self.readGauge(), missed);
            for (&sides) |*sd| if (sd.r) |r| {
                self.stream.release(r);
                sd.r = null;
            };
            return acc;
        }

        /// Side `sd`'s positions in wave `w`, grouped by bank in first appearance: each group's rows of x through the
        /// quant's fused wave (over `over`'s waited arrays when given), their positions in the call recorded in `acc`.
        fn fusedWave(self: *Self, g: *G, sd: *const Side, x: T, k: u32, w: u8, acc: *Acc, over: ?*const [n_banks]?Arrays) ![]const T {
            const first = acc.n_outs;
            var groups: [n_banks]Group = undefined;
            var ng: usize = 0;
            for (sd.waves[0..sd.n], sd.refs[0..sd.n], 0..) |wv, ref, i| {
                if (wv != w) continue;
                const gi = for (groups[0..ng], 0..) |gr, j| {
                    if (gr.bank == ref.bank) break j;
                } else blk: {
                    groups[ng] = .{ .bank = ref.bank };
                    ng += 1;
                    break :blk ng - 1;
                };
                groups[gi].pos[groups[gi].n] = @intCast(i);
                groups[gi].n += 1;
            }
            for (groups[0..ng]) |*gr| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[sd.sl][@backingInt(gr.bank)]) orelse return error.SlotArraysUnbound;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |i, *t, *r| {
                    t.* = @intCast(sd.pos[i] / k);
                    r.* = sd.refs[i].row;
                }
                const nn: c_int = @intCast(gr.n);
                const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{nn}, .int32), 0);
                acc.outs[acc.n_outs] = try self.math.fused(g, xs, rows[0..gr.n], arrays);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |i| {
                    acc.pos[acc.n_pos] = sd.pos[i];
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

        /// A decode-lane call's host split: barrier [t0, t1), routes [t1, t2), waves [t2, now).
        fn noteHost(self: *Self, t0: u64, t1: u64, t2: u64) void {
            const t3 = timeline.now();
            self.host.calls += 1;
            self.host.barrier_ns += t1 -| t0;
            self.host.route_ns += t2 -| t1;
            self.host.wave_ns += t3 -| t2;
        }

        /// The read pool's wall time with a read in flight (ns): the timeline's read gauge.
        fn readGauge(self: *const Self) u64 {
            return @intCast(@max(self.stream.pool.readGauge()[4], 0));
        }

        /// The wide lane: the weighted sum `[n, hidden]`.
        fn callWide(self: *Self, g: *G, layer: u32, x: T, indices: T, scores: T, n: u32, k: u32) !T {
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            // The staged routes this call does not take (a bank layer it routes nothing on) go back at its end.
            defer self.dropStaged();
            const tb = pt.now();
            try w.ids.resize(a, n_ids);
            _ = try g.hostIds(indices, w.ids.items);
            pt.chargeRouted(.barrier, tb);
            w.pos.clearRetainingCapacity();
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            w.window = .{};
            // x as the math's prefill reads it (`beginWide`: GLM-5.3's kernels' f16 copy, made once for the call and
            // dropped with its waves), else x.
            const copy: ?T = if (@hasDecl(M, "beginWide")) try self.math.beginWide(g, x) else null;
            defer if (copy) |c| g.drop(c);
            for (0..bpl) |side| try self.wideBank(g, layer, @intCast(side), copy orelse x, k);
            return self.wideJoin(g, scores, x, n, k);
        }

        /// The wide lane's groups on stream layer `bpl x layer + side`: the routed rows whose expert it holds, their
        /// outputs appended to `wide.kept` and their positions to `wide.pos`.
        fn wideBank(self: *Self, g: *G, layer: u32, side: u32, x: T, k: u32) !void {
            const a = self.a;
            const w = &self.wide;
            const sl = bpl * layer + side;
            // An empty bank layer: nothing routes on it.
            if (bpl > 1) if (self.stream.bank.slots[sl].empty()) return;
            const group_n = self.stream.max_route_ids;
            var tp = pt.now();
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            try w.count.resize(a, self.n_experts);
            @memset(w.count.items, 0);
            w.distinct.clearRetainingCapacity();
            w.side_ids.clearRetainingCapacity();
            for (w.ids.items) |e| {
                if (bpl > 1 and self.stream.bank.streamLayer(layer, e) != sl) continue;
                try w.side_ids.append(a, e);
                w.count.items[e] += 1;
                if (w.first.items[e] < 0) {
                    w.first.items[e] = 0;
                    try w.distinct.append(a, e);
                }
            }
            if (w.distinct.items.len == 0) return;
            // Hottest first (rows descending, ties by id): the first groups' slots are the seed's.
            std.sort.pdq(u16, w.distinct.items, @as([]const u32, w.count.items), struct {
                fn lt(cnt: []const u32, p: u16, q: u16) bool {
                    return if (cnt[p] != cnt[q]) cnt[p] > cnt[q] else p < q;
                }
            }.lt);
            pt.chargeRouted(.barrier, tp);
            tp = pt.now();
            try self.stream.awaitReadAhead(sl);
            pt.chargeRouted(.ahead, tp);
            tp = pt.now();
            try self.stream.seedPrefill(sl, w.side_ids.items);
            // The groups: the bank layer's residents (hottest first), then its misses read at the layer's start
            // (`stageMisses`, as their routes took them), then the rest (hottest first). Every resident is routed before
            // any other miss is planned, so no miss evicts a resident whose group is still to come; each expert is read
            // at most once in the call.
            const policy = &self.stream.layers[sl].policy;
            const st = &self.staged;
            const staged = st.layer != null and st.layer.? == layer;
            try w.staged_mark.resize(a, self.n_experts);
            @memset(w.staged_mark.items, false);
            // The staged routes of this bank layer, and those of the others still holding a window.
            var n_staged: usize = 0;
            var held_other: usize = 0;
            if (staged) for (0..st.n_routes) |ri| {
                if (st.routes[ri] == null) continue;
                if (st.sls[ri] != sl) {
                    held_other += 1;
                    continue;
                }
                n_staged += 1;
                for (st.experts.items[st.bounds[ri]..st.bounds[ri + 1]]) |e| w.staged_mark.items[e] = true;
            };
            w.order.clearRetainingCapacity();
            w.bounds.clearRetainingCapacity();
            try w.bounds.append(a, 0);
            var first_staged: usize = 0;
            for ([_]bool{ true, false }) |resident| {
                for (w.distinct.items) |e| if ((policy.slotOf(e) != null) == resident and !w.staged_mark.items[e]) {
                    if (w.order.items.len - w.bounds.items[w.bounds.items.len - 1] == group_n) try w.bounds.append(a, @intCast(w.order.items.len));
                    try w.order.append(a, e);
                };
                if (w.order.items.len > w.bounds.items[w.bounds.items.len - 1]) try w.bounds.append(a, @intCast(w.order.items.len));
                if (!resident) continue;
                // The staged routes' groups, between the residents' and the rest's.
                first_staged = w.bounds.items.len - 1;
                if (n_staged > 0) for (0..st.n_routes) |ri| if (st.routes[ri] != null and st.sls[ri] == sl) {
                    try w.order.appendSlice(a, st.experts.items[st.bounds[ri]..st.bounds[ri + 1]]);
                    try w.bounds.append(a, @intCast(w.order.items.len));
                };
            }
            const n_groups = w.bounds.items.len - 1;
            try w.egroup.resize(a, self.n_experts);
            for (0..n_groups) |gi| for (w.order.items[w.bounds.items[gi]..w.bounds.items[gi + 1]], w.bounds.items[gi]..) |e, i| {
                w.first.items[e] = @intCast(i);
                w.egroup.items[e] = @intCast(gi);
            };
            try w.groute.resize(a, n_groups);
            @memset(w.groute.items, null);
            // Windows held: the staged routes (this bank layer's, taken here, and the others').
            var live: usize = held_other;
            errdefer for (w.groute.items) |rt| if (rt) |r| self.stream.release(r);
            if (n_staged > 0) {
                var gs = first_staged;
                for (0..st.n_routes) |ri| if (st.routes[ri] != null and st.sls[ri] == sl) {
                    w.groute.items[gs] = st.routes[ri];
                    st.routes[ri] = null;
                    gs += 1;
                };
                live += n_staged;
            }
            // The bank layer's routed rows bucketed by group once (a counting sort, each group's rows in row order; a row
            // of another bank layer's expert has no place here).
            try w.group_at.resize(a, n_groups + 1);
            @memset(w.group_at.items, 0);
            for (w.ids.items) |e| if (w.first.items[e] >= 0) {
                w.group_at.items[@as(usize, w.egroup.items[e]) + 1] += 1;
            };
            for (1..n_groups + 1) |gi| w.group_at.items[gi] += w.group_at.items[gi - 1];
            try w.by_group.resize(a, w.side_ids.items.len);
            try w.slot.resize(a, n_groups);
            @memcpy(w.slot.items, w.group_at.items[0..n_groups]);
            for (w.ids.items, 0..) |e, row| {
                if (w.first.items[e] < 0) continue;
                const gi = w.egroup.items[e];
                w.by_group.items[w.slot.items[gi]] = @intCast(row);
                w.slot.items[gi] += 1;
            }
            const depth: usize = self.opt.wide_depth;
            // Routes ahead, in group order, while windows are free.
            var next: usize = 0;
            const ahead = struct {
                fn f(slf: *Self, wd: *Wide, lyr: u32, nxt: *usize, lv: *usize, dp: usize) !void {
                    while (nxt.* < wd.groute.items.len and (wd.groute.items[nxt.*] != null or lv.* < dp)) : (nxt.* += 1) {
                        if (wd.groute.items[nxt.*] != null) continue;
                        wd.groute.items[nxt.*] = try slf.stream.route(lyr, wd.order.items[wd.bounds.items[nxt.*]..wd.bounds.items[nxt.* + 1]], &.{});
                        lv.* += 1;
                    }
                }
            }.f;
            try ahead(self, w, sl, &next, &live, depth);
            pt.chargeRouted(.route, tp);
            // Each group's waves committed as soon as they are built; the group before it is then drained and its
            // slots go back, so the GPU runs one group while the host builds the next.
            var prev: ?struct { gi: usize, k0: usize } = null;
            for (0..n_groups) |gi| {
                const start = w.bounds.items[gi];
                if (w.groute.items[gi] == null) {
                    // No window was free for it: the group before it drains first.
                    if (prev) |pv| try self.drainGroup(g, w, sl, pv.gi, w.kept.items[pv.k0..], &next, &live, depth, ahead);
                    prev = null;
                }
                const r = w.groute.items[gi].?;
                const tw = pt.now();
                for (0..r.n_parts) |p| {
                    try self.stream.waitGu(r, @intCast(p));
                    try self.stream.waitDown(r, @intCast(p));
                }
                pt.chargeRouted(.wait, tw);
                const tc = pt.now();
                var refs: [max_route_ids]SlotRef = undefined;
                _ = self.stream.refsOf(r, &refs);
                const k0 = w.kept.items.len;
                const m = g.mark();
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    w.slot.clearRetainingCapacity();
                    w.act.clearRetainingCapacity();
                    const p0 = w.pos.items.len;
                    for (w.by_group.items[w.group_at.items[gi]..w.group_at.items[gi + 1]]) |row| {
                        const fi: usize = @intCast(w.first.items[w.ids.items[row]]);
                        const ref = refs[fi - start];
                        if (ref.bank != kind) continue;
                        try w.slot.append(a, ref.row);
                        try w.act.append(a, @intCast(row / k));
                        try w.pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0) continue;
                    const arrays = self.banks[sl][@backingInt(kind)] orelse return error.SlotArraysUnbound;
                    // The group's rows of this bank in slices (each slice's transient bounded, `group_slice_rows`), at most
                    // the slice window's in flight.
                    var s0: usize = 0;
                    while (s0 < w.slot.items.len) : (s0 += glm.group_slice_rows) {
                        const s1 = @min(s0 + glm.group_slice_rows, w.slot.items.len);
                        try w.window.admit(g, w.kept.items);
                        const y = try self.math.prefill(g, sl, x, .{ .slot = w.slot.items[s0..s1], .act_row = w.act.items[s0..s1] }, arrays);
                        try w.kept.append(a, g.keep(y));
                        try w.window.commit(g, w.kept.items, w.kept.items.len - 1);
                    }
                    std.debug.assert(w.pos.items.len - p0 == w.slot.items.len);
                }
                try self.math.finishPrefill(g);
                try g.asyncEval(w.kept.items[k0..]);
                g.resetTo(m);
                pt.chargeRouted(.encode, tc);
                if (prev) |pv| try self.drainGroup(g, w, sl, pv.gi, w.kept.items[pv.k0..k0], &next, &live, depth, ahead);
                prev = .{ .gi = gi, .k0 = k0 };
            }
            if (prev) |pv| try self.drainGroup(g, w, sl, pv.gi, w.kept.items[pv.k0..], &next, &live, depth, ahead);
        }

        /// The wide lane's join: every routed row's output in routed order, then the combine in token slices.
        fn wideJoin(self: *Self, g: *G, scores: T, x: T, n: u32, k: u32) !T {
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            try w.inv.resize(a, n_ids);
            // On the GPU the combine reads each routed row where its group's call left it (no join).
            if (comptime G == ops.MlxOps) if (self.combine_kernel) |kern| if (w.kept.items.len <= max_combine_parts) {
                const tb = pt.now();
                defer if (pt.enabled) pt.chargeRouted(.combine, tb);
                var j: usize = 0;
                for (w.kept.items, 0..) |part, pi| for (0..@intCast(g.shapeOf(part).dim(0))) |row| {
                    w.inv.items[w.pos.items[j]] = @as(u32, @intCast(pi)) << 24 | @as(u32, @intCast(row));
                    j += 1;
                };
                const loc = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
                const out = try combineOnGpu(kern, g, w.kept.items, loc, scores, n, k, self.hidden, g.dtypeOf(x));
                // The combine holds the parts until it is evaluated.
                for (w.kept.items) |o| g.release(o);
                w.kept.clearRetainingCapacity();
                w.window = .{};
                return out;
            };
            // The join: every routed row's output in routed order, then the combine in token slices.
            const tj = pt.now();
            const joined = try g.concat(w.kept.items, 0);
            try g.evalAll(&.{joined});
            pt.chargeRouted(.join, tj);
            const tb = pt.now();
            defer if (pt.enabled) pt.chargeRouted(.combine, tb);
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            w.window = .{};
            for (w.pos.items, 0..) |p, j| w.inv.items[p] = @intCast(j);
            if (comptime G == ops.MlxOps) if (self.combine_kernel) |kern| {
                const inv = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
                return combineOnGpu(kern, g, &.{joined}, inv, scores, n, k, self.hidden, g.dtypeOf(x));
            };
            var outs: std.ArrayList(T) = .empty;
            defer outs.deinit(a);
            const ts: u32 = @intCast(glm.combine_slice_tokens);
            var t0: u32 = 0;
            while (t0 < n) : (t0 += ts) {
                const t1 = @min(t0 + ts, n);
                const rows: c_int = @intCast((t1 - t0) * k);
                const ord = try g.hostArray(std.mem.sliceAsBytes(w.inv.items[t0 * k .. t1 * k]), &.{rows}, .uint32);
                const y = try g.reshape(try g.take(joined, ord, 0), &.{ @intCast(t1 - t0), @intCast(k), self.hidden });
                const s = try g.slice(scores, &.{ @intCast(t0), 0 }, &.{ @intCast(t1), @intCast(k) }, &.{ 1, 1 });
                try outs.append(a, try combine(g, y, s, g.dtypeOf(x)));
            }
            return if (outs.items.len == 1) outs.items[0] else g.concat(outs.items, 0);
        }
    };
}

test "glm experts: the wide lane's GPU combine over its parts equals the op chain's bits at GLM-5.3's hidden size" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const set = try trunk.openForTest(a, s);
    defer set.deinit();
    const n: u32 = 300;
    const k: u32 = glm.routed_top_k;
    const hidden: c_int = 6144;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    const yv = try a.alloc(f32, n * k * @as(usize, @intCast(hidden)));
    defer a.free(yv);
    for (yv) |*v| v.* = rand.floatNorm(f32);
    const wv = try a.alloc(f32, n * k);
    defer a.free(wv);
    for (wv) |*v| v.* = rand.float(f32);
    // The joined rows in a shuffled order and its inverse, as the wide lane's groups leave them.
    const inv = try a.alloc(u32, n * k);
    defer a.free(inv);
    for (inv, 0..) |*v, i| v.* = @intCast(i);
    rand.shuffle(u32, inv);
    const joined = try g.astype(try g.hostArray(std.mem.sliceAsBytes(yv), &.{ @intCast(n * k), hidden }, .float32), .bfloat16);
    const scores = try g.hostArray(std.mem.sliceAsBytes(wv), &.{ @intCast(n), @intCast(k) }, .float32);
    const ord = try g.hostArray(std.mem.sliceAsBytes(inv), &.{@intCast(n * k)}, .uint32);
    // The rows in 3 parts, each row's place as (part, row).
    const cuts = [_]c_int{ 0, 700, 1900, @intCast(n * k) };
    var parts: [3]ops.MlxOps.T = undefined;
    for (&parts, 0..) |*p, i| p.* = try g.slice(joined, &.{ cuts[i], 0 }, &.{ cuts[i + 1], hidden }, &.{ 1, 1 });
    const locv = try a.alloc(u32, n * k);
    defer a.free(locv);
    for (inv, locv) |r, *l| {
        const pi: u32 = if (r < cuts[1]) 0 else if (r < cuts[2]) 1 else 2;
        l.* = pi << 24 | (r - @as(u32, @intCast(cuts[pi])));
    }
    const loc = try g.hostArray(std.mem.sliceAsBytes(locv), &.{@intCast(n * k)}, .uint32);
    const got = try combineOnGpu(&set.bound, &g, &parts, loc, scores, n, k, hidden, .bfloat16);
    const y = try g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), hidden });
    const want = try g.astype(try g.sum(try g.mul(y, try g.expandDims(scores, -1)), -2, false), .bfloat16);
    const same = try g.astype(try g.equal(got, want), .float32);
    const all = try g.sum(try g.sum(same, -1, false), -1, false);
    var out: [1]f32 = undefined;
    _ = try g.hostF32(all, &out);
    try std.testing.expectEqual(@as(f32, @floatFromInt(n * @as(u32, @intCast(hidden)))), out[0]);
}
