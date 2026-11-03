//! Per-layer expert residency policy of the expert streamer. Pure: it plans
//! which slot serves each routed expert and which records must be read, and
//! performs neither. A port of the Python stack's LayerExpertSlotBank in the
//! tier's configuration: one request, one resident pool per layer plus a
//! shared transient scratch, 2Q prefill admission behind the prompt-frequency
//! seed, transition-window decode admission. That class is the oracle
//! (the reference runtime's dump_phase1_route_fixture.py replays a recorded trace
//! through it for the parity test below).

const std = @import("std");

pub const Phase = enum { prefill, decode };

/// Widest route a plan holds: 8 verify rows x top-6 routed experts.
pub const max_route_ids = 48;
pub const no_expert: u16 = std.math.maxInt(u16);
pub const no_slot: u32 = std.math.maxInt(u32);

/// Decode admission: routes remembered by the frequency term, and the weights of
/// the transition prediction, window frequency and recency terms.
const window_limit = 16;
const w_prediction: f32 = 0.7;
const w_frequency: f32 = 0.2;
const w_recency: f32 = 0.1;

pub const Load = struct { expert: u16, slot: u32, persistent: bool };
pub const Eviction = struct { slot: u32, previous: u16, next: u16 };

/// One route's decisions, in the orders of the Python RoutePlan.
pub const Plan = struct {
    phase: Phase = .decode,
    n_ids: u32 = 0,
    /// Per routed id, its slot: persistent slots are [0, capacity), the
    /// transient scratch follows them.
    slots: [max_route_ids]u32 = undefined,
    n_hits: u32 = 0,
    /// Resident experts, in first-appearance order.
    hits: [max_route_ids]u16 = undefined,
    n_misses: u32 = 0,
    /// Non-resident experts, in admission order.
    misses: [max_route_ids]u16 = undefined,
    n_loads: u32 = 0,
    /// Persistent loads in admission order, then transient loads.
    loads: [max_route_ids]Load = undefined,
    /// The persistent loads: `loads[0..n_persistent]`.
    n_persistent: u32 = 0,
    n_evictions: u32 = 0,
    evictions: [max_route_ids]Eviction = undefined,

    pub fn slotsOf(p: *const Plan) []const u32 {
        return p.slots[0..p.n_ids];
    }
    pub fn hitsOf(p: *const Plan) []const u16 {
        return p.hits[0..p.n_hits];
    }
    pub fn missesOf(p: *const Plan) []const u16 {
        return p.misses[0..p.n_misses];
    }
    pub fn loadsOf(p: *const Plan) []const Load {
        return p.loads[0..p.n_loads];
    }
    pub fn evictionsOf(p: *const Plan) []const Eviction {
        return p.evictions[0..p.n_evictions];
    }
};

pub const LayerPolicy = struct {
    n_experts: u32,
    /// Persistent slots; raised once, at the decode transition.
    capacity: u32,
    occupancy: u32 = 0,
    /// [n_experts]: slots beyond `capacity` stay empty.
    slot_to_expert: []u16,
    expert_to_slot: []u32,
    // Prefill: 2Q single pool behind the prompt-frequency seed.
    prefill_freq: []u32,
    seed: std.DynamicBitSetUnmanaged,
    protected: std.DynamicBitSetUnmanaged,
    /// The last `prepareSeed` call's routed rows per expert (the read-ahead's barrier tally reads them).
    call_counts: []u32,
    /// The last `prepareSeed`'s chosen ranks: the call's hottest `seed_ranks` experts (count desc, ties by id).
    seed_ranks: u32 = 0,
    /// Pool clock stamp of each resident expert (0 = none).
    recency: []u64,
    clock: u64 = 0,
    // Decode: one-step transitions + a bounded route window.
    epoch: i64 = 0,
    last_used: []i64,
    /// [n_experts * n_experts]: transitions previous route -> current route.
    counts: []f32,
    denominators: []f32,
    window_freq: []f32,
    window: [window_limit][max_route_ids]u16 = undefined,
    window_len: [window_limit]u8 = undefined,
    window_head: u32 = 0,
    window_count: u32 = 0,
    // Per-plan scratch.
    in_route: std.DynamicBitSetUnmanaged,
    retained: std.DynamicBitSetUnmanaged,
    scores: []f32,
    candidates: []u16,
    victims: []u32,
    available: []u32,
    admission: []u32,
    /// Persistent slots another live route of this layer still serves from
    /// (`PlanOpts.held`, set for one plan): never a victim.
    held: std.DynamicBitSetUnmanaged,
    // The rate tally (`setTally`, `sdk_ext.expert.tally`): per expert an EMA of "a decode route of this layer requests
    // it" (`rate`, `alpha` = 1 - 2^(-1/window) a route), folded in at each request's end from the request's counts
    // (`endRequest`: a decode route only counts its experts). Once seeded (a tally file read, or a request ended) the
    // top `pin_limit` experts by rate, at most the capacity, are pinned for the next request: never a victim, first in
    // the decode cut, and the only prefill misses that may evict. Off (no `setTally`) or unseeded: the plans are those
    // without a tally.
    rate: []f32,
    pinned: std.DynamicBitSetUnmanaged,
    /// This request's decode routes, and per expert the routes that requested it (`endRequest`).
    request_counts: []u32,
    request_routes: u32 = 0,
    alpha: f32 = 0,
    /// Whether decode routes count into `request_counts`: the tally's (`setTally`) and the usage profile's
    /// (`expert.usage`, `Stream.countRoutes`); set once at construction.
    count_routes: bool = false,
    pin_limit: u32 = 0,
    seeded: bool = false,
    pin_scratch: []u16,

    /// A plan beside other live routes of the layer: their slots (`held`) are
    /// never victims, and its transient loads take rows from `transient_base`.
    /// `held` is a prefill plan's: a decode plan takes none (the stream's phase
    /// change leaves decode one window and no held base).
    pub const PlanOpts = struct { transient_base: u32 = 0, held: []const u32 = &.{} };

    pub fn init(a: std.mem.Allocator, n_experts: u32, capacity: u32) !LayerPolicy {
        if (n_experts == 0 or n_experts >= no_expert or capacity > n_experts) return error.InvalidCapacity;
        const n: usize = n_experts;
        var p: LayerPolicy = undefined;
        p = .{
            .n_experts = n_experts,
            .capacity = capacity,
            .slot_to_expert = try a.alloc(u16, n),
            .expert_to_slot = undefined,
            .prefill_freq = undefined,
            .seed = undefined,
            .protected = undefined,
            .call_counts = undefined,
            .recency = undefined,
            .last_used = undefined,
            .counts = undefined,
            .denominators = undefined,
            .window_freq = undefined,
            .in_route = undefined,
            .retained = undefined,
            .scores = undefined,
            .candidates = undefined,
            .victims = undefined,
            .available = undefined,
            .admission = undefined,
            .held = undefined,
            .rate = undefined,
            .pinned = undefined,
            .request_counts = undefined,
            .pin_scratch = undefined,
        };
        errdefer a.free(p.slot_to_expert);
        p.expert_to_slot = try a.alloc(u32, n);
        errdefer a.free(p.expert_to_slot);
        p.prefill_freq = try a.alloc(u32, n);
        errdefer a.free(p.prefill_freq);
        p.seed = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.seed.deinit(a);
        p.protected = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.protected.deinit(a);
        p.call_counts = try a.alloc(u32, n);
        errdefer a.free(p.call_counts);
        p.recency = try a.alloc(u64, n);
        errdefer a.free(p.recency);
        p.last_used = try a.alloc(i64, n);
        errdefer a.free(p.last_used);
        p.counts = try a.alloc(f32, n * n);
        errdefer a.free(p.counts);
        p.denominators = try a.alloc(f32, n);
        errdefer a.free(p.denominators);
        p.window_freq = try a.alloc(f32, n);
        errdefer a.free(p.window_freq);
        p.in_route = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.in_route.deinit(a);
        p.retained = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.retained.deinit(a);
        p.scores = try a.alloc(f32, n);
        errdefer a.free(p.scores);
        p.candidates = try a.alloc(u16, n + max_route_ids);
        errdefer a.free(p.candidates);
        p.victims = try a.alloc(u32, n);
        errdefer a.free(p.victims);
        p.available = try a.alloc(u32, n);
        errdefer a.free(p.available);
        p.admission = try a.alloc(u32, n);
        errdefer a.free(p.admission);
        p.held = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.held.deinit(a);
        p.rate = try a.alloc(f32, n);
        errdefer a.free(p.rate);
        p.pinned = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.pinned.deinit(a);
        p.request_counts = try a.alloc(u32, n);
        errdefer a.free(p.request_counts);
        p.pin_scratch = try a.alloc(u16, n);
        @memset(p.rate, 0);
        @memset(p.request_counts, 0);
        @memset(p.slot_to_expert, no_expert);
        @memset(p.expert_to_slot, no_slot);
        @memset(p.prefill_freq, 0);
        @memset(p.call_counts, 0);
        @memset(p.recency, 0);
        @memset(p.last_used, -1);
        @memset(p.counts, 0);
        @memset(p.denominators, 0);
        @memset(p.window_freq, 0);
        @memset(p.admission, no_slot);
        return p;
    }

    pub fn deinit(p: *LayerPolicy, a: std.mem.Allocator) void {
        a.free(p.slot_to_expert);
        a.free(p.expert_to_slot);
        a.free(p.prefill_freq);
        p.seed.deinit(a);
        p.protected.deinit(a);
        a.free(p.call_counts);
        a.free(p.recency);
        a.free(p.last_used);
        a.free(p.counts);
        a.free(p.denominators);
        a.free(p.window_freq);
        p.in_route.deinit(a);
        p.retained.deinit(a);
        a.free(p.scores);
        a.free(p.candidates);
        a.free(p.victims);
        a.free(p.available);
        a.free(p.admission);
        p.held.deinit(a);
        a.free(p.rate);
        p.pinned.deinit(a);
        a.free(p.request_counts);
        a.free(p.pin_scratch);
        p.* = undefined;
    }

    pub fn slotOf(p: *const LayerPolicy, expert: u16) ?u32 {
        const s = p.expert_to_slot[expert];
        return if (s == no_slot) null else s;
    }

    /// The rate tally on: at most `pin_limit` experts pinned (the ratio's share of the layer's experts), the rate an EMA
    /// of factor `alpha` per decode route. Unseeded until `seedTally` or `endRequest`.
    pub fn setTally(p: *LayerPolicy, pin_limit: u32, alpha: f32) void {
        p.pin_limit = @min(pin_limit, p.n_experts);
        p.alpha = alpha;
        p.count_routes = true;
    }

    /// The rates a tally file (or another layer's state) holds, one per expert: the tally seeded, the pins chosen.
    pub fn seedTally(p: *LayerPolicy, rates: []const f32) void {
        std.debug.assert(rates.len == p.n_experts);
        for (p.rate, rates) |*r, x| r.* = if (x > 0 and x <= 1) x else 0;
        p.seeded = true;
        p.refreshPins();
    }

    /// The request's end: `out` (one per expert) set to this request's decode rate (each expert's share of the layer's
    /// decode routes) where the layer routed in its decode; the rate folded in over the request's routes (R of them:
    /// rate x (1 - alpha)^R + (1 - (1 - alpha)^R) x the share, the per-route EMA with the request's routes taken as
    /// one uniform run); the request's counts zeroed; the pins chosen again, the tally seeded from now on (a later
    /// request on this load pins). Returns whether the layer routed in the request's decode.
    pub fn endRequest(p: *LayerPolicy, out: []f32) bool {
        std.debug.assert(out.len == p.n_experts);
        if (p.alpha == 0) return false;
        const routes = p.request_routes;
        if (routes > 0) {
            const n: f32 = @floatFromInt(routes);
            const decay = std.math.pow(f32, 1 - p.alpha, n);
            for (out, p.request_counts, p.rate) |*o, c, *r| {
                o.* = @as(f32, @floatFromInt(c)) / n;
                r.* = r.* * decay + (1 - decay) * o.*;
            }
        }
        @memset(p.request_counts, 0);
        p.request_routes = 0;
        p.seeded = true;
        p.refreshPins();
        return routes > 0;
    }

    /// The pins at the tally's rates: the top `min(pin_limit, capacity)` experts of positive rate, ties by the lower id.
    /// None while unseeded.
    pub fn refreshPins(p: *LayerPolicy) void {
        p.pinned.unsetAll();
        const k = @min(p.pin_limit, p.capacity);
        if (!p.seeded or k == 0) return;
        var n: usize = 0;
        for (p.rate, 0..) |r, e| if (r > 0) {
            p.pin_scratch[n] = @intCast(e);
            n += 1;
        };
        const ranked = p.pin_scratch[0..n];
        std.sort.pdq(u16, ranked, @as([]const f32, p.rate), struct {
            fn hotter(rs: []const f32, x: u16, y: u16) bool {
                return if (rs[x] != rs[y]) rs[x] > rs[y] else x < y;
            }
        }.hotter);
        for (ranked[0..@min(k, n)]) |e| p.pinned.set(e);
    }

    pub fn isPinned(p: *const LayerPolicy, e: u16) bool {
        return p.pinned.isSet(e);
    }

    /// One decode route of the layer (its unique experts) into the request's counts: the rate takes them at the
    /// request's end (`endRequest`), not per route.
    fn tallyRoute(p: *LayerPolicy, unique: []const u16) void {
        if (!p.count_routes) return;
        for (unique) |e| p.request_counts[e] += 1;
        p.request_routes += 1;
    }

    /// With the tally seeded, a prefill miss that is not pinned takes only an empty row (it evicts nothing, and no seed
    /// promotion copies it): the residents the tally's preload and pins chose stay.
    fn mayEvictInPrefill(p: *const LayerPolicy, e: u16) bool {
        return !p.seeded or p.pin_limit == 0 or p.pinned.isSet(e);
    }

    /// prepare_prefill_seed: counts the prompt's routed ids, then re-protects
    /// the resident part of its top-(capacity - protected) and marks the rest to
    /// be admitted first, protected, by the prefill routes.
    pub fn prepareSeed(p: *LayerPolicy, ids: []const u16) void {
        const counts = p.call_counts;
        @memset(counts, 0);
        for (ids) |e| {
            p.prefill_freq[e] += 1;
            counts[e] += 1;
        }
        const empty: i64 = @as(i64, p.capacity) - @as(i64, @intCast(p.protected.count()));
        p.seed.unsetAll();
        p.seed_ranks = 0;
        if (empty <= 0) return;
        const ranked = rankHottest(counts, p.candidates);
        const chosen = ranked[0..@min(ranked.len, @as(usize, @intCast(empty)))];
        p.seed_ranks = @intCast(chosen.len);
        // Resident choices: re-protected in ascending count order (stable).
        var n_res: usize = 0;
        for (chosen) |e| if (p.expert_to_slot[e] != no_slot) {
            p.victims[n_res] = e;
            n_res += 1;
        };
        const res = p.victims[0..n_res];
        std.sort.insertion(u32, res, @as([]const u32, counts), struct {
            fn lessThan(c: []const u32, x: u32, y: u32) bool {
                return c[x] < c[y];
            }
        }.lessThan);
        for (res) |e| {
            p.protected.set(e);
            p.clock += 1;
            p.recency[e] = p.clock;
        }
        for (chosen) |e| if (p.expert_to_slot[e] == no_slot and p.mayEvictInPrefill(e)) p.seed.set(e);
    }

    /// Forgets every resident and the prompt state (protection, the seed, the prompt counts, recency): what the
    /// construction's warm-up leaves belongs to no prompt. Called once, nothing live; returns the residents forgotten.
    pub fn forgetAll(p: *LayerPolicy) u32 {
        const n = p.occupancy;
        for (p.slot_to_expert) |*e| if (e.* != no_expert) {
            p.expert_to_slot[e.*] = no_slot;
            e.* = no_expert;
        };
        p.occupancy = 0;
        // No resident left: forgetPrompt clears the prompt state and sets no recency.
        _ = p.forgetPrompt();
        return n;
    }

    /// `forgetAll` without the residents: before a continuation of the same conversation the residents stay in their
    /// slots and only the prompt state goes (protection, the seed, the prompt counts). Each resident's recency becomes its
    /// last decode use (never used: 0), so the prompt's first victims are the residents decode used least recently.
    /// Called once, nothing live, after `shrink`; returns the residents kept.
    pub fn forgetPrompt(p: *LayerPolicy) u32 {
        p.protected.unsetAll();
        p.seed.unsetAll();
        p.seed_ranks = 0;
        @memset(p.prefill_freq, 0);
        @memset(p.recency, 0);
        p.clock = 0;
        for (p.slot_to_expert[0..p.capacity]) |e| if (e != no_expert and p.last_used[e] >= 0) {
            p.recency[e] = @intCast(p.last_used[e] + 1);
            p.clock = @max(p.clock, p.recency[e]);
        };
        return p.occupancy;
    }

    /// The one phase change: `capacity` persistent slots from now on, the new
    /// ones empty.
    pub fn grow(p: *LayerPolicy, capacity: u32) !void {
        if (capacity < p.capacity or capacity > p.n_experts) return error.InvalidCapacity;
        p.capacity = capacity;
    }

    /// The return to the prompt phase before a later prompt (the served path's per-request cycle: the rows past
    /// `capacity` are freed for the prompt pass's waves and grown back at its phase change): `capacity`
    /// persistent slots from now on, and every expert resident in a slot at or past it forgotten, so no plan
    /// serves from a freed row.
    pub fn shrink(p: *LayerPolicy, capacity: u32) !void {
        if (capacity > p.capacity) return error.InvalidCapacity;
        var s: u32 = capacity;
        while (s < p.capacity) : (s += 1) {
            const e = p.slot_to_expert[s];
            if (e != no_expert) p.invalidate(e);
        }
        p.capacity = capacity;
    }

    /// One read-ahead admission (`admitReadAhead`): the expert and the persistent slot it took.
    pub const ReadAhead = struct { expert: u16, slot: u32 };

    /// P1's read-ahead: each predicted expert (hottest first) not resident takes an empty persistent slot, else the slot
    /// of the coldest resident the prediction left out (lowest (recency, slot); never a protected, pinned or held one, nor
    /// one a plan is routing), unprotected, until no slot is free to take; `prepareSeed` then re-protects those its seed
    /// chooses (hits), the rest stay evictable. A resident the prediction names is never a victim: it is a hit to be.
    /// `held`: slots live routes of the layer still serve from (never victims). Returns the admissions (at most `out.len`).
    pub fn admitReadAhead(p: *LayerPolicy, experts: []const u16, out: []ReadAhead) []ReadAhead {
        return p.admitReadAheadHeld(experts, &.{}, out);
    }

    pub fn admitReadAheadHeld(p: *LayerPolicy, experts: []const u16, held: []const u32, out: []ReadAhead) []ReadAhead {
        for (held) |s| p.held.set(s);
        defer for (held) |s| p.held.unset(s);
        for (experts) |e| if (e < p.n_experts) p.retained.set(e);
        defer for (experts) |e| if (e < p.n_experts) p.retained.unset(e);
        var n: usize = 0;
        for (experts) |e| {
            if (n == out.len) break;
            if (e >= p.n_experts or p.expert_to_slot[e] != no_slot) continue;
            const slot = p.emptySlot() orelse p.aheadVictim() orelse break;
            const victim = p.slot_to_expert[slot];
            if (victim != no_expert) {
                p.expert_to_slot[victim] = no_slot;
                p.protected.unset(victim);
                p.recency[victim] = 0;
            } else p.occupancy += 1;
            p.slot_to_expert[slot] = e;
            p.expert_to_slot[e] = slot;
            p.clock += 1;
            p.recency[e] = p.clock;
            out[n] = .{ .expert = e, .slot = slot };
            n += 1;
        }
        return out[0..n];
    }

    /// The read-ahead's victim: the coldest probationary resident (`probationVictim`'s rule) the prediction did not
    /// name (`retained` holds the predicted experts for the admission).
    fn aheadVictim(p: *const LayerPolicy) ?u32 {
        var best: ?u32 = null;
        var best_recency: u64 = 0;
        for (p.slot_to_expert[0..p.capacity], 0..) |e, s| {
            if (e == no_expert or p.retained.isSet(e) or p.in_route.isSet(e) or p.protected.isSet(e) or p.held.isSet(s) or p.pinned.isSet(e)) continue;
            if (best == null or p.recency[e] < best_recency) {
                best = @intCast(s);
                best_recency = p.recency[e];
            }
        }
        return best;
    }

    /// Forgets a resident expert (its record failed to load).
    pub fn invalidate(p: *LayerPolicy, expert: u16) void {
        const s = p.expert_to_slot[expert];
        if (s == no_slot) return;
        p.slot_to_expert[s] = no_expert;
        p.expert_to_slot[expert] = no_slot;
        p.occupancy -= 1;
        p.protected.unset(expert);
        p.recency[expert] = 0;
    }

    /// Resolves one route. `ids` holds at most `max_route_ids` expert ids
    /// (< n_experts, repeats allowed: M rows x top-k); the transient scratch
    /// must hold `max_route_ids` slots.
    pub fn plan(p: *LayerPolicy, ids: []const u16, phase: Phase, out: *Plan) void {
        p.planWith(ids, phase, out, .{});
    }

    pub fn planWith(p: *LayerPolicy, ids: []const u16, phase: Phase, out: *Plan, opts: PlanOpts) void {
        std.debug.assert(ids.len > 0 and ids.len <= max_route_ids);
        std.debug.assert(phase == .prefill or opts.held.len == 0);
        for (opts.held) |s| p.held.set(s);
        defer for (opts.held) |s| p.held.unset(s);
        out.* = .{ .phase = phase, .n_ids = @intCast(ids.len) };
        var unique_buf: [max_route_ids]u16 = undefined;
        var n_unique: usize = 0;
        for (ids) |e| {
            std.debug.assert(e < p.n_experts);
            if (p.in_route.isSet(e)) continue;
            p.in_route.set(e);
            unique_buf[n_unique] = e;
            n_unique += 1;
        }
        const unique = unique_buf[0..n_unique];
        defer for (unique) |e| p.in_route.unset(e);

        if (phase == .decode) {
            p.epoch += 1;
            for (ids) |e| p.last_used[e] = p.epoch;
            p.observe(unique);
            p.tallyRoute(unique);
        }
        for (unique) |e| {
            if (p.expert_to_slot[e] != no_slot) {
                out.hits[out.n_hits] = e;
                out.n_hits += 1;
            } else {
                out.misses[out.n_misses] = e;
                out.n_misses += 1;
            }
        }
        // Prefill hits refresh recency; decode hits touch only transition state.
        // (Python walks a set here: the order among one wave's hits can differ.)
        if (phase == .prefill) for (out.hitsOf()) |e| {
            p.clock += 1;
            p.recency[e] = p.clock;
        };

        var transient_buf: [max_route_ids]u16 = undefined;
        var n_transient: usize = 0;
        switch (phase) {
            .decode => {
                const admissions = p.transitionAdmissions(out);
                for (out.missesOf()) |e| {
                    const slot = admissions[e];
                    if (slot == no_slot) {
                        transient_buf[n_transient] = e;
                        n_transient += 1;
                        continue;
                    }
                    p.assign(slot, e, out);
                    out.loads[out.n_loads] = .{ .expert = e, .slot = slot, .persistent = true };
                    out.n_loads += 1;
                }
                for (out.missesOf()) |e| admissions[e] = no_slot;
            },
            .prefill => {
                p.seedFirst(out);
                // `in_route` holds this route's experts: its hits and each miss
                // once admitted are never victims.
                for (out.missesOf()) |e| {
                    const is_seed = p.seed.isSet(e);
                    const slot = p.emptySlot() orelse (if (p.mayEvictInPrefill(e)) p.probationVictim() else null) orelse no_slot;
                    if (is_seed) p.seed.unset(e);
                    if (slot == no_slot) {
                        transient_buf[n_transient] = e;
                        n_transient += 1;
                        continue;
                    }
                    const victim = p.slot_to_expert[slot];
                    if (victim != no_expert) {
                        p.protected.unset(victim);
                        p.recency[victim] = 0;
                    }
                    p.assign(slot, e, out);
                    p.clock += 1;
                    p.recency[e] = p.clock;
                    p.protected.setValue(e, is_seed);
                    out.loads[out.n_loads] = .{ .expert = e, .slot = slot, .persistent = true };
                    out.n_loads += 1;
                }
            },
        }
        out.n_persistent = out.n_loads;
        for (transient_buf[0..n_transient], 0..) |e, k| {
            out.loads[out.n_loads] = .{ .expert = e, .slot = p.capacity + opts.transient_base + @as(u32, @intCast(k)), .persistent = false };
            out.n_loads += 1;
        }
        for (ids, 0..) |e, i| out.slots[i] = p.slotFor(e, out);
    }

    fn slotFor(p: *const LayerPolicy, e: u16, out: *const Plan) u32 {
        if (p.expert_to_slot[e] != no_slot) return p.expert_to_slot[e];
        for (out.loadsOf()) |l| if (l.expert == e) return l.slot;
        unreachable;
    }

    fn assign(p: *LayerPolicy, slot: u32, e: u16, out: *Plan) void {
        const previous = p.slot_to_expert[slot];
        if (previous != no_expert) {
            p.expert_to_slot[previous] = no_slot;
            out.evictions[out.n_evictions] = .{ .slot = slot, .previous = previous, .next = e };
            out.n_evictions += 1;
        } else p.occupancy += 1;
        p.slot_to_expert[slot] = e;
        p.expert_to_slot[e] = slot;
    }

    fn emptySlot(p: *const LayerPolicy) ?u32 {
        if (p.occupancy >= p.capacity) return null;
        for (p.slot_to_expert[0..p.capacity], 0..) |e, s| if (e == no_expert) return @intCast(s);
        return null;
    }

    /// Prefill: seed expert `e`, not resident (its record read into a transient row before the seed was known),
    /// admitted as a miss of a plan would be: an empty slot, else the coldest probationary resident's, none of the
    /// slots in `held`; protected. Returns its slot (the caller copies the record there), or null where no slot is free
    /// to take.
    pub fn admitSeed(p: *LayerPolicy, e: u16, held: []const u32) ?u32 {
        std.debug.assert(p.seed.isSet(e) and p.expert_to_slot[e] == no_slot);
        for (held) |s| p.held.set(s);
        defer for (held) |s| p.held.unset(s);
        const slot = p.emptySlot() orelse p.probationVictim() orelse return null;
        const victim = p.slot_to_expert[slot];
        if (victim != no_expert) {
            p.expert_to_slot[victim] = no_slot;
            p.protected.unset(victim);
            p.recency[victim] = 0;
        } else p.occupancy += 1;
        p.slot_to_expert[slot] = e;
        p.expert_to_slot[e] = slot;
        p.seed.unset(e);
        p.clock += 1;
        p.recency[e] = p.clock;
        p.protected.set(e);
        return slot;
    }

    /// The coldest probationary resident (lowest (recency, slot)) that this
    /// route does not hold; prefill never evicts a protected expert.
    fn probationVictim(p: *const LayerPolicy) ?u32 {
        var best: ?u32 = null;
        var best_recency: u64 = 0;
        for (p.slot_to_expert[0..p.capacity], 0..) |e, s| {
            if (e == no_expert or p.in_route.isSet(e) or p.protected.isSet(e) or p.held.isSet(s) or p.pinned.isSet(e)) continue;
            if (best == null or p.recency[e] < best_recency) {
                best = @intCast(s);
                best_recency = p.recency[e];
            }
        }
        return best;
    }

    /// Seed misses first, least frequent first (stable), then the rest.
    fn seedFirst(p: *const LayerPolicy, out: *Plan) void {
        var seeds: [max_route_ids]u16 = undefined;
        var rest: [max_route_ids]u16 = undefined;
        var ns: usize = 0;
        var nr: usize = 0;
        for (out.missesOf()) |e| {
            if (p.seed.isSet(e)) {
                seeds[ns] = e;
                ns += 1;
            } else {
                rest[nr] = e;
                nr += 1;
            }
        }
        if (ns == 0) return;
        std.sort.insertion(u16, seeds[0..ns], @as([]const u32, p.prefill_freq), struct {
            fn lessThan(f: []const u32, x: u16, y: u16) bool {
                return f[x] < f[y];
            }
        }.lessThan);
        @memcpy(out.misses[0..ns], seeds[0..ns]);
        @memcpy(out.misses[ns..][0..nr], rest[0..nr]);
    }

    /// Publishes one decode route (its unique experts) into the transition
    /// counts and the route window.
    fn observe(p: *LayerPolicy, current: []const u16) void {
        const n = p.n_experts;
        if (p.window_count > 0) {
            const prev_i = (p.window_head + p.window_count - 1) % window_limit;
            const prev = p.window[prev_i][0..p.window_len[prev_i]];
            for (prev) |r| {
                for (current) |c| p.counts[@as(usize, r) * n + c] += 1.0;
                p.denominators[r] += @floatFromInt(current.len);
            }
        }
        if (p.window_count == window_limit) {
            const old = p.window[p.window_head][0..p.window_len[p.window_head]];
            for (old) |e| p.window_freq[e] -= 1.0;
            p.window_head = (p.window_head + 1) % window_limit;
            p.window_count -= 1;
        }
        const at = (p.window_head + p.window_count) % window_limit;
        @memcpy(p.window[at][0..current.len], current);
        p.window_len[at] = @intCast(current.len);
        p.window_count += 1;
        for (current) |e| p.window_freq[e] += 1.0;
    }

    /// The causal decode scores (float32, in the Python evaluation order):
    /// 0.7 * transition prediction from the current route + 0.2 * window
    /// frequency / its max + 0.1 * 1 / (1 + epoch - last use).
    fn computeScores(p: *LayerPolicy) void {
        const n = p.n_experts;
        const cur_i = (p.window_head + p.window_count - 1) % window_limit;
        const current = p.window[cur_i][0..p.window_len[cur_i]];
        @memset(p.scores, 0);
        for (current) |r| {
            const d = p.denominators[r];
            if (!(d > 0)) continue;
            const row = p.counts[@as(usize, r) * n ..][0..n];
            for (p.scores, row) |*s, c| s.* += c / d;
        }
        var max_window: f32 = 1.0;
        for (p.window_freq) |f| max_window = @max(max_window, f);
        for (p.scores, 0..) |*s, e| {
            const lu = p.last_used[e];
            const recency: f32 = if (lu >= 0) @floatCast(1.0 / (1.0 + @as(f64, @floatFromInt(p.epoch - lu)))) else 0;
            s.* = w_prediction * s.* + w_frequency * (p.window_freq[e] / max_window) + w_recency * recency;
        }
    }

    /// (pinned, score, last use, prompt frequency, -expert): Python's rank tuple behind the tally's pins (none unseeded).
    fn rankLess(p: *const LayerPolicy, a: u16, b: u16) bool {
        const pa = p.pinned.isSet(a);
        if (pa != p.pinned.isSet(b)) return !pa;
        if (p.scores[a] != p.scores[b]) return p.scores[a] < p.scores[b];
        if (p.last_used[a] != p.last_used[b]) return p.last_used[a] < p.last_used[b];
        if (p.prefill_freq[a] != p.prefill_freq[b]) return p.prefill_freq[a] < p.prefill_freq[b];
        return a > b;
    }

    /// One bounded cut over residents and misses: of everything that may
    /// change (empty slots + residents this route does not hold), keep the
    /// highest ranks. Returns `admission[expert]` = its persistent slot, or
    /// no_slot (served transient); callers reset the misses' entries.
    fn transitionAdmissions(p: *LayerPolicy, out: *const Plan) []u32 {
        const admission = p.admission;
        if (out.n_misses == 0) return admission;
        p.computeScores();
        const cap = p.capacity;
        var n_cand: usize = 0;
        var n_evictable: usize = 0;
        for (p.slot_to_expert[0..cap]) |e| {
            if (e == no_expert or p.in_route.isSet(e)) continue;
            p.candidates[n_cand] = e;
            n_cand += 1;
            n_evictable += 1;
        }
        const free_budget = cap - p.occupancy;
        var n_empty: usize = 0;
        for (p.slot_to_expert[0..cap], 0..) |e, s| {
            if (n_empty == free_budget) break;
            if (e == no_expert) {
                p.available[n_empty] = @intCast(s);
                n_empty += 1;
            }
        }
        const adjustable = n_empty + n_evictable;
        if (adjustable == 0) return admission;
        for (out.missesOf()) |e| {
            p.candidates[n_cand] = e;
            n_cand += 1;
        }
        const keep = @min(adjustable, n_cand);
        const cands = p.candidates[0..n_cand];
        std.sort.pdq(u16, cands, @as(*const LayerPolicy, p), struct {
            fn greater(pp: *const LayerPolicy, a: u16, b: u16) bool {
                return pp.rankLess(b, a);
            }
        }.greater);
        for (cands[0..keep]) |e| p.retained.set(e);
        defer for (cands[0..keep]) |e| p.retained.unset(e);

        var n_victims: usize = 0;
        for (p.slot_to_expert[0..cap], 0..) |e, s| {
            if (e == no_expert or p.in_route.isSet(e) or p.retained.isSet(e)) continue;
            p.victims[n_victims] = @intCast(s);
            n_victims += 1;
        }
        std.sort.pdq(u32, p.victims[0..n_victims], @as(*const LayerPolicy, p), struct {
            fn less(pp: *const LayerPolicy, a: u32, b: u32) bool {
                return pp.rankLess(pp.slot_to_expert[a], pp.slot_to_expert[b]);
            }
        }.less);
        @memcpy(p.available[n_empty..][0..n_victims], p.victims[0..n_victims]);
        var next: usize = 0;
        for (out.missesOf()) |e| {
            if (!p.retained.isSet(e)) continue;
            admission[e] = p.available[next];
            next += 1;
        }
        return admission;
    }
};

/// Python's _bounded_decode_miss_route_parts over records already in
/// placement order: parts of at most `per_part`, each cut moved back to the
/// last physical gap inside its window so a contiguous run stays whole.
/// Writes each part's end index; returns them.
/// The seed's ranking (`prepareSeed`; P1's predicted seed): every expert counted at least once, count
/// descending, ties by id. A strict total order, so the result is a function of `counts` alone.
pub fn rankHottest(counts: []const u32, out: []u16) []u16 {
    var n: usize = 0;
    for (counts, 0..) |c, e| if (c > 0) {
        out[n] = @intCast(e);
        n += 1;
    };
    std.sort.pdq(u16, out[0..n], counts, struct {
        fn lessThan(cs: []const u32, x: u16, y: u16) bool {
            return if (cs[x] != cs[y]) cs[x] > cs[y] else x < y;
        }
    }.lessThan);
    return out[0..n];
}

pub fn boundedParts(offsets: []const u64, lengths: []const u64, per_part: u32, ends: []u32) []u32 {
    const n = offsets.len;
    var n_parts: usize = 0;
    var start: usize = 0;
    while (start < n) {
        var end = @min(start + per_part, n);
        if (end < n) {
            var gap: ?usize = null;
            var i = start + 1;
            while (i <= end) : (i += 1) {
                if (offsets[i - 1] + lengths[i - 1] != offsets[i]) gap = i;
            }
            if (gap) |g| end = g;
        }
        ends[n_parts] = @intCast(end);
        n_parts += 1;
        start = end;
    }
    return ends[0..n_parts];
}
