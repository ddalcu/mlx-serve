//! Thinking Machines Inkling 2 Mini (`inkling2_loop`, text and images), in mlx-lm's layout (`models/inkling2`) or as released:
//! 24 layers of weights applied 48 times per token, the middle 12 run three times, each pass with its own KV cache and conv
//! windows and, after the first, messages from the pass before. The stream between layers is depth attention over
//! saved blocks, not a plain residual, so the forward is the plugin's own loop (`layers.zig`), not a base's.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const device = @import("../../mlx/device.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const block = @import("../../sdk/bases/block.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const loader = @import("../llama/load.zig");
const layers = @import("layers.zig");
const kernels = @import("kernels.zig");
const vision = @import("vision.zig");
const media = @import("media.zig");
const draft = @import("draft.zig");
const kv = @import("../../sdk/bases/kv.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const gpa = std.heap.c_allocator;

/// MLX's nvfp4 global scale is the block's amax; the codes times their block scales carry it over 6 x 448.
const nvfp4_global_unit: f32 = 6.0 * 448.0;

/// What the bundled drafter reads off the trunk's forwards: the taps (the stream entering a few visits, and the
/// final-normed output) at the newest committed positions of a sequence. The ring is one engine slot's (`select`);
/// the other slots that ran lately keep theirs parked, so interleaved requests each draft from their own context.
/// A forward that does not continue its slot's ring restarts it.
pub const Spec = struct {
    /// The visits whose input is a tap; the last tap is the final-normed output.
    visits: []const u32,
    /// Rows kept: the drafter's window less the block's anchor.
    keep: c_int,
    /// `[1, n, taps * hidden]` and the position after its last row.
    ring: Array = .{},
    end: u64 = 0,
    /// The drafter's context keys and values projected from the ring, parked and released with it.
    context: draft.ContextKv = .{},
    /// The engine slot the ring is of, and the rings of other slots, the least recently parked replaced first.
    owner: u32 = 0,
    parked: [3]Parked = @splat(.{}),
    clock: u64 = 0,
    /// Set for a verify: the visits read the caches without appending and leave their rows here.
    opens: ?[]kernels.Open = null,
    /// A verify's taps `[1, S, taps * hidden]`, held until its accept.
    held: Array = .{},
    /// Set when a drafter layer attends every committed position: a push hands it each forward's rows, all of them.
    full: ?*const draft.Drafter = null,

    const Parked = struct { owner: u32 = 0, ring: Array = .{}, end: u64 = 0, at: u64 = 0, context: draft.ContextKv = .{} };

    /// Makes `slot`'s ring the current one.
    pub fn select(sp: *Spec, slot: u32) void {
        if (slot == sp.owner) return;
        sp.clock += 1;
        var to: usize = 0;
        var found = false;
        for (sp.parked, 0..) |p, i| {
            if (p.ring.ctx != null and p.owner == slot) {
                to = i;
                found = true;
                break;
            }
            if (p.at < sp.parked[to].at) to = i;
        }
        var theirs = sp.parked[to];
        if (!found) {
            if (theirs.ring.ctx != null) _ = mlx.mlx_array_free(theirs.ring);
            theirs.context.release();
        }
        sp.parked[to] = .{ .owner = sp.owner, .ring = sp.ring, .end = sp.end, .at = sp.clock, .context = sp.context };
        sp.ring = if (found) theirs.ring else .{};
        sp.end = if (found) theirs.end else 0;
        sp.context = if (found) theirs.context else .{};
        sp.owner = slot;
    }

    /// The context rows before `position`; 0 when the ring ends elsewhere.
    pub fn rows(sp: *const Spec, position: u64) c_int {
        return if (sp.ring.ctx != null and sp.end == position) ops.shape(sp.ring)[1] else 0;
    }

    /// `taps` are the rows from `first` on: they extend the ring or restart it.
    fn push(sp: *Spec, g: *ops.Graph, first: u64, taps: ops.Array) !void {
        const n = ops.shape(taps)[1];
        const continues = sp.rows(first) > 0;
        if (!continues) sp.context.release();
        if (sp.full) |d| try d.appendFull(g, &sp.context, taps, first);
        var all = if (continues) try g.concat(&.{ sp.ring, taps }, 1) else taps;
        const total = ops.shape(all)[1];
        if (total > sp.keep) all = try g.contiguous(try g.sliceAxis(all, 1, total - sp.keep, total));
        const next = g.escape(all);
        // Evaluated now: left lazy, each forward would hang another concat off the last.
        const vec = mlx.mlx_vector_array_new_value(next);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
        if (sp.ring.ctx != null) _ = mlx.mlx_array_free(sp.ring);
        sp.ring = next;
        sp.end = first + @as(u64, @intCast(n));
    }

    fn deinit(sp: *Spec) void {
        for ([_]Array{ sp.ring, sp.held, sp.parked[0].ring, sp.parked[1].ring, sp.parked[2].ring }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        sp.context.release();
        for (&sp.parked) |*p| p.context.release();
    }
};

pub const Model = struct {
    pub const batches_decode = true;

    pub const Weights = struct {
        embed: b.embed.TableEmbed.Weights,
        embed_norm: Array,
        final_norm: Array,
        head: layers.Head.Weights,
        body: layers.Body,
        /// The drafter's taps; null on a pack without one.
        spec: ?*Spec = null,
    };

    /// One forward of `ids` (`[1, rows]` int32). Results are owned by `g`.
    pub fn forward(w: *const Weights, shared: *const block.Shared, g: *ops.Graph, ids: ops.Array, run: run_mod.Run) !run_mod.Out {
        var c = run.ctx(g, shared);
        // The embedding is normed before the media rows land on it, as mlx-lm's `embed_inputs`.
        const h = try run.input(g, try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &c, ids), w.embed_norm, w.body.cfg.rms_norm_eps), ids);
        if (run.slots == 1) return sequence(w, shared, g, h, run);
        // Batched slots, a row each, ride the kernels as rows; elsewhere (a CPU stream, several rows) a slot at a time.
        if (run.rows == 1 and kernels.batches(g, &w.body, h, run.slots)) {
            var out = try sequence(w, shared, g, try g.reshape(h, &.{ 1, run.slots, -1 }), run);
            if (out.hidden.ctx != null) out.hidden = try g.reshape(out.hidden, &.{ run.slots, 1, -1 });
            return out;
        }
        const n: usize = @intCast(run.slots);
        if (n > kernels.max_slots) return error.TooManySlots;
        var logits: [kernels.max_slots]ops.Array = undefined;
        var hidden: [kernels.max_slots]ops.Array = undefined;
        for (0..n) |i| {
            const at: c_int = @intCast(i);
            var one = run;
            one.slots = 1;
            one.positions = run.positions[i .. i + 1];
            one.caches = run.caches[i .. i + 1];
            const out = try sequence(w, shared, g, try g.sliceAxis(h, 0, at, at + 1), one);
            logits[i] = out.logits;
            hidden[i] = out.hidden;
        }
        return .{
            .logits = if (logits[0].ctx != null) try g.concat(logits[0..n], 0) else .{},
            .hidden = if (hidden[0].ctx != null) try g.concat(hidden[0..n], 0) else .{},
        };
    }

    /// The forward over the embedded rows `h_in`: one sequence's `[1, S, D]`, or batched slots as rows `[1, N, D]`.
    fn sequence(w: *const Weights, shared: *const block.Shared, g: *ops.Graph, h_in: ops.Array, run: run_mod.Run) !run_mod.Out {
        const cfg = &w.body.cfg;
        var c = run.ctx(g, shared);
        var h = h_in;
        const batched = run.slots > 1;

        // The open depth blocks, stacked `[1, rows, n_blocks, hidden]`.
        var blocks: ?ops.Array = null;
        var n_blocks: c_int = 0;
        // A later pass restarts from the loop's input and reads the pass before through `bank` and `bank_partial`.
        var loop_in: ops.Array = .{};
        var loop_stack: ?ops.Array = null;
        var loop_blocks: c_int = 0;
        var bank: ops.Array = .{};
        var bank_partial: ops.Array = .{};

        const visits = cfg.visits();
        const spec: ?*Spec = if (batched) null else w.spec;
        var taps: [config.max_taps]ops.Array = undefined;
        for (0..visits) |v| {
            const at = cfg.visit(@intCast(v));
            if (cfg.core() > 0 and at.layer == cfg.core_start) {
                if (at.pass == 0) {
                    loop_in = h;
                    loop_stack = blocks;
                    loop_blocks = n_blocks;
                } else {
                    bank = try g.sliceAxis(blocks.?, 2, loop_blocks + 1, n_blocks);
                    bank_partial = try layers.passOutput(g, &w.body, h);
                    if (h.ctx != loop_in.ctx) g.drop(h);
                    h = loop_in;
                    blocks = loop_stack;
                    n_blocks = loop_blocks;
                }
            }
            if (spec) |sp| for (sp.visits, 0..) |tap, k| if (tap == v) {
                taps[k] = try g.keep(h);
            };
            var lg = g.scope();
            defer lg.deinit();
            c.g = &lg;
            c.layer = @intCast(v);
            const L = &w.body.layers[at.layer];
            const messaged = at.pass > 0;
            const open: ?*kernels.Open = if (spec) |sp| (if (sp.opens) |o| &o[v] else null) else null;
            const stepped: ?layers.Post = try kernels.visit(&c, L, &w.body, h, blocks, if (messaged) .{ bank, bank_partial } else null, open);
            // A verify's rows must not reach the caches, nor batched slots the one-sequence chain.
            if (stepped == null and (open != null or batched)) return error.Inkling2KernelVisit;
            const out = stepped orelse try layers.layerForward(L, &w.body, &c, h, blocks, if (messaged) try layers.messagesFor(&lg, L, bank, bank_partial) else null);
            c.g = g;
            const old = h;
            h = try g.keep(out.h);
            if (out.blocks) |stack| {
                blocks = try g.keep(stack);
                n_blocks += 1;
            }
            if (old.ctx != loop_in.ctx) g.drop(old);
            try run.capture(g, @intCast(v), h);
            try run.cadence(v, visits, h);
        }
        const stepped: ?ops.Array = try kernels.finalStream(g, &w.body, h, blocks, w.final_norm);
        var final = stepped orelse try g.astype(try g.rmsNorm(try layers.finalInput(g, &w.body, h, blocks), w.final_norm, cfg.rms_norm_eps), ops.dtype(h));
        if (spec) |sp| {
            // The newest rows' taps, side by side; a verify's wait for its accept.
            taps[sp.visits.len] = final;
            const from = if (sp.full != null) 0 else @max(0, run.rows - sp.keep);
            var parts: [config.max_taps]ops.Array = undefined;
            for (taps[0 .. sp.visits.len + 1], parts[0 .. sp.visits.len + 1]) |x, *part| part.* = try g.sliceAxis(x, 1, from, run.rows);
            const rows = try g.concat(parts[0 .. sp.visits.len + 1], 2);
            if (sp.opens != null) {
                if (sp.held.ctx != null) _ = mlx.mlx_array_free(sp.held);
                sp.held = g.escape(rows);
            } else try sp.push(g, run.positions[0] + @as(u64, @intCast(from)), rows);
        }
        if (batched) final = try g.reshape(final, &.{ run.slots, 1, -1 });
        return run.finish(g, &c, final, layers.Head, &w.head);
    }
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = config.layout },
    .weights = &.{},
    .media = &.{ .tower = &media.tower, .prepare = &media.prepare },
    .spec = &.{ .native_drafter = &.{ .step = &draftStep, .verify = &draftVerify, .finish = &draftFinish, .active = &draftActive } },
};

fn draftStep(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    round(m, f, out) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

fn draftVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    verifyRound(m, f, out) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

fn draftActive(mp: *anyopaque) callconv(.c) bool {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return m.drafter != null;
}

fn draftFinish(mp: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    finishVerify(m, accepted, out) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

/// The trunk's pick per row of `logits` `[rows, vocab]` under the sampler's suppress mask `[vocab]` bool: `[rows]`.
fn picks(g: *ops.Graph, logits: ops.Array, suppress: abi.Array) !ops.Array {
    const masked = if (suppress.ctx == null) logits else try g.where(.{ .ctx = suppress.ctx }, try g.scalar(-std.math.inf(f32), .float32), logits);
    return g.astype(try g.argmax(masked, -1, false), .int32);
}

/// What a verified row costs against a plain step on this machine, from the rounds' own timings (ms): the drafts a
/// round verifies end where the drafter's probability of the run so far falls under `cut` (a cheaper row is worth a
/// less sure draft).
pub const RowCost = struct {
    /// The least of eight samples: a plain step's spread from boot to boot moved the cut by a few hundredths.
    plain: Cost = .{ .seeds = 8 },
    /// Logged once the cut is first measured.
    told: bool = false,
    /// Each draft count's verify, 1 to `max_round_tokens - 1`.
    verify: [abi.max_round_tokens]Cost = @splat(.{}),

    /// A running cost; the least of its first `seeds` samples seeds it (a first launch compiles).
    const Cost = struct {
        ms: f32 = 0,
        seen: u8 = 0,
        seeds: u8 = 3,

        pub fn add(c: *Cost, ms: f32) void {
            if (!std.math.isFinite(ms) or ms <= 0) return;
            if (c.seen < c.seeds) {
                c.ms = if (c.seen == 0) ms else @min(c.ms, ms);
                c.seen += 1;
                return;
            }
            c.ms += 0.1 * (std.math.clamp(ms, 0.8 * c.ms, 1.25 * c.ms) - c.ms);
        }
    };

    /// A verified row over a plain step: the least-squares slope of verify cost over the draft counts seen, over the
    /// plain step's cost; null until a plain step and two counts are.
    fn ratio(self: *const RowCost) ?f32 {
        if (self.plain.seen == 0) return null;
        var n: f32 = 0;
        var sk: f32 = 0;
        var sy: f32 = 0;
        var skk: f32 = 0;
        var sky: f32 = 0;
        for (self.verify[1..], 1..) |c, k| if (c.seen > 0) {
            const x: f32 = @floatFromInt(k);
            n += 1;
            sk += x;
            sy += c.ms;
            skk += x * x;
            sky += x * c.ms;
        };
        if (n < 2) return null;
        return (n * sky - sk * sy) / (n * skk - sk * sk) / self.plain.ms;
    }

    /// The run probability a round verifies drafts down to: 1.34 sqrt(ratio), the curve through the best cut measured
    /// per machine (the verify-cut test pins them); 0.5 until measured.
    pub fn cut(self: *const RowCost) f32 {
        const r = self.ratio() orelse return 0.5;
        return std.math.clamp(1.34 * @sqrt(@max(r, 0)), 0.3, 0.8);
    }
};

/// A round's open verify (or its plain step): the drafter proposed after the committed token `t1` and the trunk read `t1`
/// and the drafts it was sure of as one forward that appends nothing, its graph keeping what a commit reads (each
/// visit's rows in `opens`, the taps in `Spec.held`). With no context, or no draft worth a row, the forward was a plain
/// step that appended `t1` already.
const Opened = struct {
    g: *ops.Graph,
    fo: abi.ForwardOut = .{},
    opens: []kernels.Open = &.{},
    drafts: u32 = 0,
    /// `t1`, then the drafts.
    ids: [abi.max_round_tokens]u32 = undefined,
    position: [1]u32,
    cache: ?abi.Handle = null,
    cache_ops: ?*const abi.CacheOps = null,
    started: i64 = 0,

    fn deinit(o: *Opened) void {
        for ([_]abi.Array{ o.fo.logits, o.fo.hidden }) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
        };
        gpa.free(o.opens);
        o.g.deinit();
        gpa.destroy(o.g);
    }
};

/// Drafts after `f`'s token and reads the block (or steps plainly): `fo.logits` are its rows'.
fn openRound(m: *Loaded, f: *const abi.Forward) !Opened {
    const w = &m.base.weights;
    const cfg = &w.body.cfg;
    const t1 = f.tokens.slice()[0];
    const position = f.positions.slice()[0];
    const g = try gpa.create(ops.Graph);
    g.* = ops.Graph.init(gpa, m.base.s);
    var o: Opened = .{ .g = g, .position = .{position} };
    errdefer o.deinit();
    o.ids[0] = t1;
    if (f.caches.len == 1) o.cache = f.caches.slice()[0];
    o.cache_ops = f.cache_ops;

    const sp = w.spec;
    if (sp) |x| if (f.slots.len == 1) x.select(f.slots.slice()[0]);
    const context: c_int = if (sp) |x| x.rows(position) else 0;
    const cap: u32 = if (m.drafter) |d| @min(f.accept_cap, d.cfg.block - 1) else 0;
    var proposed: ops.Array = .{};
    const first = try g.fromSlice(i32, &.{@intCast(t1)}, &.{1});
    if (cap > 0 and context > 0 and f.cache_ops != null and f.caches.len == 1) {
        const r: run_mod.Run = .{ .mode = .verify, .rows = 1, .positions = &o.position, .caches = f.caches.slice(), .cache_ops = f.cache_ops };
        var c = r.ctx(g, &m.base.shared);
        const anchor = try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &c, try g.reshape(first, &.{ 1, 1 })), w.embed_norm, cfg.rms_norm_eps);
        const drafted = try m.drafter.?.draft(g, first, anchor, sp.?.ring, @intCast(position), &w.head.lin, w.head.vocab, w.head.softcap, &sp.?.context);
        try ops.evalMany(&.{ drafted.tokens, drafted.confidence });
        const sure = mlx.mlx_array_data_float32(drafted.confidence) orelse return error.Inkling2DraftRound;
        const tokens = mlx.mlx_array_data_int32(drafted.tokens) orelse return error.Inkling2DraftRound;
        const min_sure = m.row_cost.cut();
        if (!m.row_cost.told) if (m.row_cost.ratio()) |ratio| {
            m.row_cost.told = true;
            log.info("[inkling2] draft verify cut {d:.2} (a verified row {d:.2} of a plain step)\n", .{ min_sure, ratio });
        };
        const stops: []const u32 = if (f.size >= @offsetOf(abi.Forward, "stop_ids") + @sizeOf(abi.Span(u32))) f.stop_ids.slice() else &.{};
        o.drafts = draftsToVerify(sure[0..cap], tokens[0..cap], min_sure, stops);
        for (0..o.drafts) |k| o.ids[1 + k] = @intCast(tokens[k]);
        proposed = try g.sliceAxis(drafted.tokens, 0, 0, @intCast(o.drafts));
    }

    o.started = nowUs();
    if (o.drafts == 0) {
        const v = run_mod.tableForward(Model, &m.base, &.{ .mode = .decode, .logits = .last_row, .rows = 1, .tokens = f.tokens, .slots = f.slots, .positions = f.positions, .caches = f.caches, .cache_ops = f.cache_ops }, &o.fo);
        if (!v.isOk()) return error.Inkling2DraftRound;
        return o;
    }
    o.opens = try gpa.alloc(kernels.Open, cfg.visits());
    @memset(o.opens, .{ .g = g });
    sp.?.opens = o.opens;
    defer sp.?.opens = null;
    const v = run_mod.tableForward(Model, &m.base, &.{ .mode = .verify, .logits = .all_rows, .rows = 1 + o.drafts, .tokens = .{}, .ids = .{ .ctx = (try g.concat(&.{ first, proposed }, 0)).ctx }, .slots = f.slots, .positions = f.positions, .caches = f.caches, .cache_ops = f.cache_ops }, &o.fo);
    if (!v.isOk()) return error.Inkling2DraftRound;
    return o;
}

/// The drafts a round verifies: while their running probability stays at `min_sure` or above, and none after a stop
/// id (`Forward.stop_ids`), which ends the turn.
fn draftsToVerify(sure: []const f32, tokens: []const i32, min_sure: f32, stops: []const u32) u32 {
    var run: f32 = 1;
    var n: u32 = 0;
    while (n < sure.len) : (n += 1) {
        run *= sure[n];
        if (run < min_sure or std.mem.indexOfScalar(u32, stops, @intCast(tokens[n])) != null) break;
    }
    return n;
}

/// Keeps `t1` and the first `accepted` drafts of `o`: their keys, values and conv windows committed, as the serial
/// steps would have left them, and their taps on the ring. The position after them.
fn commitOpen(m: *Loaded, o: *Opened, accepted: u32) !u64 {
    const position: u64 = o.position[0];
    if (o.drafts == 0) return position + 1;
    if (accepted > o.drafts) return error.Inkling2DraftRound;
    const sp = m.base.weights.spec.?;
    const kept: c_int = @intCast(1 + accepted);
    const handles = [_]abi.Handle{o.cache orelse return error.Inkling2DraftRound};
    const r: run_mod.Run = .{ .mode = .verify, .rows = kept, .positions = &o.position, .caches = &handles, .cache_ops = o.cache_ops };
    var c = r.ctx(o.g, &m.base.shared);
    try kernels.commit(&c, &m.base.weights.body.cfg, o.opens, kept);
    try sp.push(o.g, position, try o.g.sliceAxis(sp.held, 1, 0, kept));
    return position + @as(u64, @intCast(kept));
}

/// One greedy draft round on the committed token `t1` (`NativeDrafter.step`): the rows whose pick is the next draft
/// are kept. With no context yet, or no draft worth a row, it is a plain step.
fn round(m: *Loaded, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    var o = try openRound(m, f);
    defer o.deinit();
    const picked = try picks(o.g, .{ .ctx = o.fo.logits.ctx }, f.suppress_mask);
    try ops.eval(picked);
    const want = mlx.mlx_array_data_int32(picked) orelse return error.Inkling2DraftRound;
    if (o.drafts == 0) m.row_cost.plain.add(msSince(o.started)) else m.row_cost.verify[o.drafts].add(msSince(o.started));
    var accepted: u32 = 0;
    while (accepted < o.drafts and want[accepted] == @as(i32, @intCast(o.ids[1 + accepted]))) accepted += 1;
    const position = try commitOpen(m, &o, accepted);
    out.round = .{ .n = 1 + accepted, .accepted = accepted, .next_token = @intCast(want[accepted]), .rows = 1 + o.drafts, .position = position };
    @memcpy(out.round.tokens[0 .. 1 + accepted], o.ids[0 .. 1 + accepted]);
}

/// The first half of a sampled round (`NativeDrafter.verify`): the open verify's rows' ids and logits `[rows, vocab]`
/// f32 for the engine to accept, the verify kept open until `finishVerify` (a new one discards it).
fn verifyRound(m: *Loaded, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    if (m.opened) |x| {
        x.deinit();
        gpa.destroy(x);
        m.opened = null;
    }
    const o = try gpa.create(Opened);
    errdefer gpa.destroy(o);
    o.* = try openRound(m, f);
    errdefer o.deinit();
    const n: c_int = @intCast(1 + o.drafts);
    const logits = try o.g.astype(try o.g.reshape(.{ .ctx = o.fo.logits.ctx }, &.{ n, -1 }), .float32);
    try ops.eval(logits);
    if (o.drafts == 0) m.row_cost.plain.add(msSince(o.started)) else m.row_cost.verify[o.drafts].add(msSince(o.started));
    out.logits = .{ .ctx = o.g.escape(logits).ctx };
    out.round = .{ .n = 1 + o.drafts, .rows = 1 + o.drafts, .position = o.position[0] };
    @memcpy(out.round.tokens[0 .. 1 + o.drafts], o.ids[0 .. 1 + o.drafts]);
    m.opened = o;
}

/// The second half (`NativeDrafter.finish`): keeps `t1` and the first `accepted` drafts of the open verify.
fn finishVerify(m: *Loaded, accepted: u32, out: *abi.ForwardOut) !void {
    const o = m.opened orelse return error.Inkling2NoOpenVerify;
    m.opened = null;
    defer {
        o.deinit();
        gpa.destroy(o);
    }
    out.round.position = try commitOpen(m, o, accepted);
}

fn nowUs() i64 {
    return @import("../../sdk/util/io_util.zig").nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

fn msSince(t0: i64) f32 {
    return @as(f32, @floatFromInt(nowUs() - t0)) / 1000;
}

/// The engine's default arithmetic over the shell and the layout, plus the drafter's context (`draftStatePerToken`).
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{ .state_per_token = config.draftStatePerToken(p) };
}

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []layers.Layer = &.{},
    owned: loader.Owned = .{},
    /// The image tower and merger; null on a pack shipped without them.
    tower: ?vision.Tower = null,
    /// The bundled drafter and the taps it reads; null on a pack shipped without one.
    drafter: ?*draft.Drafter = null,
    /// The cost a verified row adds here, which sets how sure a draft must be to be verified.
    row_cost: RowCost = .{},
    /// A sampled round's verify, open until its finish.
    opened: ?*Opened = null,
    spec: Spec = .{ .visits = &.{}, .keep = 0 },
    tap_visits: [config.max_taps]u32 = @splat(0),

    fn release(m: *Loaded) void {
        if (m.tower) |*t| t.deinit();
        if (m.opened) |o| {
            o.deinit();
            gpa.destroy(o);
        }
        if (m.drafter) |d| d.deinit();
        m.spec.deinit();
        m.owned.release();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (f.slots.len == 1) m.spec.select(f.slots.slice()[0]);
    return run_mod.tableForward(Model, &m.base, f, out);
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

/// What tensors are read with: the pack, what the model owns and the layer. Each read names the tensor twice: in
/// mlx-lm's layout, then as the release publishes it (`release`: thinkingmachines' own names under `model.llm`,
/// which mlx-lm renames at load).
const At = struct {
    r: *loader.Reader,
    o: *loader.Owned,
    layer: ?u32 = null,
    release: bool,

    fn name(a: At, buf: []u8, comptime leaf: []const u8, comptime rel: []const u8) ![]const u8 {
        if (a.release) return if (a.layer) |i| a.r.name(buf, "layers.{d}." ++ rel, .{i}) else a.r.name(buf, rel, .{});
        return if (a.layer) |i| a.r.name(buf, "layers.{d}." ++ leaf, .{i}) else a.r.name(buf, leaf, .{});
    }

    fn need(a: At, comptime leaf: []const u8, comptime rel: []const u8) !Array {
        var buf: [256]u8 = undefined;
        const x = try a.r.need(try a.name(&buf, leaf, rel));
        try a.o.own(x);
        return x;
    }

    /// The tensor evaluated as f32 and reshaped (null = as stored).
    fn float(a: At, comptime leaf: []const u8, comptime rel: []const u8, shape: ?[]const c_int) !Array {
        var buf: [256]u8 = undefined;
        const raw = try a.r.need(try a.name(&buf, leaf, rel));
        defer _ = mlx.mlx_array_free(raw);
        return a.owned32(raw, shape, 1);
    }

    /// A depth query (`rows` of them) as the forward holds it, `[hidden, rows]` f32 times hidden^-0.5: a layer's own
    /// tensor in mlx-lm's layout; entry `index` of `attn_res.<rel>`, or `attn_res.<rel>` itself (`index` null), in
    /// the release.
    fn query(a: At, comptime leaf: []const u8, comptime rel: []const u8, index: ?u32, rows: c_int, hidden: c_int) !Array {
        var buf: [256]u8 = undefined;
        const full = if (!a.release)
            try a.name(&buf, leaf, "")
        else if (index) |i|
            try a.r.name(&buf, "attn_res." ++ rel ++ ".{d}", .{i})
        else
            try a.r.name(&buf, "attn_res." ++ rel, .{});
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        const raw = try g.adopt(try a.r.need(full));
        return a.depthQuery(&g, try g.reshape(try g.astype(raw, .float32), &.{ 1, rows, hidden }));
    }

    /// Queries `[N, rows, hidden]` f32 as `[N, hidden, rows]` times hidden^-0.5 (one array when N is 1).
    fn depthQuery(a: At, g: *ops.Graph, q: Array) !Array {
        const sh = ops.shape(q);
        const scaled = try g.mulScalar(try g.contiguous(try g.transpose(q, &.{ 0, 2, 1 })), 1.0 / @sqrt(@as(f32, @floatFromInt(sh[2]))));
        return a.keep(g, if (sh[0] == 1) try g.reshape(scaled, &.{ sh[2], sh[1] }) else scaled);
    }

    fn owned32(a: At, raw: Array, shape: ?[]const c_int, times: f32) !Array {
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        var x = try g.astype(raw, .float32);
        if (shape) |sh| x = try g.reshape(x, sh);
        if (times != 1) x = try g.mulScalar(x, times);
        return a.keep(&g, x);
    }

    /// `x` evaluated, owned by the model.
    fn keep(a: At, g: *ops.Graph, x: Array) !Array {
        try ops.eval(x);
        const out = g.escape(x);
        try a.o.own(out);
        return out;
    }

    /// A projection in the checkpoint's layout (`[out, in]` when dense).
    fn proj(a: At, comptime leaf: []const u8, comptime rel: []const u8, in_dim: u32) !Linear {
        const l = try a.projRaw(leaf, rel, in_dim);
        try a.o.linear(l);
        return l;
    }

    /// The tensor with its axes permuted, evaluated.
    fn permuted(a: At, comptime leaf: []const u8, comptime rel: []const u8, axes: []const c_int) !Array {
        var buf: [256]u8 = undefined;
        const raw = try a.r.need(try a.name(&buf, leaf, rel));
        defer _ = mlx.mlx_array_free(raw);
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        return a.keep(&g, try g.transpose(raw, axes));
    }

    /// A depthwise conv kernel `[C, K, 1]`; the release stores `[C, 1, K]`.
    fn kernel(a: At, comptime leaf: []const u8, comptime rel: []const u8) !Array {
        return if (a.release) a.permuted(leaf, rel, &.{ 0, 2, 1 }) else a.need(leaf, rel);
    }

    /// A projection the caller owns once it has tried a join.
    fn projRaw(a: At, comptime leaf: []const u8, comptime rel: []const u8, in_dim: u32) !Linear {
        var buf: [256]u8 = undefined;
        return a.r.tableAt(try a.name(&buf, leaf, rel), in_dim);
    }

    /// Dense projections off one input joined along their rows: the parts become row views of the joined weight, as
    /// `loader.join` leaves a quantized group.
    fn joinRows(a: At, parts: []const *Linear) !b.linear.Joined {
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        var ws: [4]Array = undefined;
        var j: b.linear.Joined = .{};
        for (parts, 0..) |p, i| {
            ws[i] = p.w;
            j.widths[i] = ops.shape(p.w)[0];
        }
        j.lin = .{ .w = try a.keep(&g, try g.contiguous(try g.concat(ws[0..parts.len], 0))) };
        var start: c_int = 0;
        for (parts, j.widths[0..parts.len]) |p, n| {
            _ = mlx.mlx_array_free(p.w);
            p.w = g.escape(try g.sliceAxis(j.lin.w, 0, start, start + n));
            start += n;
        }
        return j;
    }

    fn conv(a: At, comptime x: []const u8, global: bool, hidden: u32) !layers.Conv {
        const leaf = "self_attn." ++ x ++ "_sconv";
        const rel = "attn." ++ x ++ "_sconv";
        if (!global) return .{ .static = try a.kernel(leaf ++ ".conv1d.weight", rel ++ ".weight") };
        return .{ .dynamic = .{
            .static = try a.need(leaf ++ ".static_weight", rel ++ ".static_w"),
            // The release's is a bare `[out, in]` tensor, with no `.weight` leaf for the format to find.
            .proj = if (a.release) .{ .w = try a.need("", rel ++ ".weight_proj") } else try a.proj(leaf ++ ".weight_proj", "", hidden),
        } };
    }

    /// An expert bank `[E, out, in]`. Neither layout names a quant scheme for the experts: a bank with e8m0/e4m3
    /// block scales is the release's own NVFP4 or MXFP8, told apart by its geometry as mlx-lm's sanitize does, with
    /// NVFP4's per-expert global scale as a multiplier in `scale` (the amax over the release's denominator).
    fn bank(a: At, comptime leaf: []const u8, comptime rel: []const u8, in_dim: u32, scale: *Array) !b.moe.Bank {
        var buf: [256]u8 = undefined;
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        var w: Array = undefined;
        var scales: Array = undefined;
        if (a.release) {
            const raw_w = try g.adopt(try a.r.need(try a.name(&buf, "", rel ++ "_weight")));
            const raw_s = try g.adopt(try a.r.need(try a.name(&buf, "", rel ++ "_weight_scale")));
            w = try a.keep(&g, try g.call(mlx.mlx_view, .{ raw_w, .uint32 }));
            scales = try a.keep(&g, try g.call(mlx.mlx_view, .{ raw_s, .uint8 }));
            const amax = try a.r.share(try a.name(&buf, "", rel ++ "_weight_amax"));
            if (amax.ctx != null) {
                const den = try g.adopt(try a.r.need(try a.name(&buf, "", rel ++ "_weight_outer_scale_denominator")));
                // In the converter's order of operations, so both layouts hold the same multiplier bit for bit.
                const global = try g.mul(try g.astype(try g.adopt(amax), .float32), try g.div(try g.scalar(nvfp4_global_unit, .float32), try g.astype(den, .float32)));
                scale.* = try a.keep(&g, try g.mulScalar(global, 1.0 / nvfp4_global_unit));
            }
        } else {
            scales = try a.r.share(try a.name(&buf, leaf ++ ".scales", ""));
            if (scales.ctx == null or mlx.mlx_array_dtype(scales) != .uint8) {
                _ = mlx.mlx_array_free(scales);
                const bk = try a.r.bank(try a.name(&buf, leaf, ""), in_dim);
                for ([_]Array{ bk.w, bk.s, bk.b }) |x| try a.o.own(x);
                return bk;
            }
            try a.o.own(scales);
            w = try a.need(leaf ++ ".weight", "");
            const gs = try a.r.share(try a.name(&buf, leaf ++ ".global_scale", ""));
            if (gs.ctx != null) {
                defer _ = mlx.mlx_array_free(gs);
                scale.* = try a.owned32(gs, null, 1.0 / nvfp4_global_unit);
            }
        }
        const bits = @as(u32, @intCast(ops.shape(w)[2])) * 32 / in_dim;
        const group = in_dim / @as(u32, @intCast(ops.shape(scales)[2]));
        var bk: b.moe.Bank = .{ .w = w, .s = scales, .qp = .{ .bits = bits, .group_size = group } };
        bk.qp.mode = if (bits == 4 and group == 16) .nvfp4 else if (bits == 4 and group == 32) .mxfp4 else if (bits == 8 and group == 32) .mxfp8 else return error.Inkling2ExpertQuantLayout;
        return bk;
    }

    /// A looped layer's four message gates, stacked.
    fn gates(a: At) !layers.Gates {
        var buf: [256]u8 = undefined;
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        var down: [4]Array = undefined;
        var forget: [4]Array = undefined;
        var message: [4]Array = undefined;
        inline for (.{ "q", "k", "v", "mlp" }, .{ "attn", "k", "v", "mlp" }, 0..) |x, rel, i| {
            const leaf = x ++ "_message_gate";
            const ctl = "controllers." ++ rel;
            down[i] = try g.expandDims(try g.adopt(try a.r.need(try a.name(&buf, leaf ++ ".down_proj.weight", ctl ++ ".down_weight"))), 0);
            forget[i] = try g.expandDims(try g.adopt(try a.r.need(try a.name(&buf, leaf ++ ".forget_gate_proj.weight", ctl ++ ".forget_weight"))), 0);
            message[i] = try g.expandDims(try g.adopt(try a.r.need(try a.name(&buf, leaf ++ ".message_gate_proj.weight", ctl ++ ".gain_weight"))), 0);
        }
        return .{ .down = try a.keep(&g, try g.concat(&down, 0)), .forget = try a.keep(&g, try g.concat(&forget, 0)), .message = try a.keep(&g, try g.concat(&message, 0)) };
    }

    /// Looped layer `index`'s q, k, v and mlp message queries as a depth query `[hidden, 4]`; the release holds
    /// `[query, mlp]` and `[key, value]` rows per looped layer in two tensors.
    fn messageQuery(at: At, core: c_int, hidden: c_int, index: c_int) !Array {
        // The queries are the model's, not a layer's.
        const a: At = .{ .r = at.r, .o = at.o, .release = at.release };
        var buf: [256]u8 = undefined;
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        if (!a.release) {
            const raw = try g.adopt(try a.r.need(try a.name(&buf, "message_input.query", "")));
            return a.depthQuery(&g, try g.sliceAxis(try g.reshape(try g.astype(raw, .float32), &.{ core, 4, hidden }), 0, index, index + 1));
        }
        var parts: [2]Array = undefined;
        inline for (.{ "middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries" }, &parts) |leaf, *p| {
            const raw = try g.adopt(try a.r.need(try a.name(&buf, "", leaf)));
            p.* = try g.sliceAxis(try g.reshape(try g.astype(raw, .float32), &.{ core, 2, hidden }), 0, index, index + 1);
        }
        return a.depthQuery(&g, try g.concat(&.{ try g.sliceAxis(parts[0], 1, 0, 1), parts[1], try g.sliceAxis(parts[0], 1, 1, 2) }, 1));
    }
};

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var release = false;
    var r = loader.Reader.open(ctx, c.block_json, &.{"model.language_model"}, "embed_tokens.weight") catch |e| switch (e) {
        error.MissingWeight => blk: {
            release = true;
            // oMLX's oQ packs name the same tensors `model.embed`, `model.layers.N.attn.in_proj`, ...
            break :blk loader.Reader.open(ctx, c.block_json, &.{"model.llm"}, "embed.weight") catch |e2| return switch (e2) {
                error.MissingWeight => error.Inkling2PackLayoutUnknown,
                else => e2,
            };
        },
        else => return e,
    };
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    m.layers = try gpa.alloc(layers.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(.{ .r = &r, .o = &m.owned, .layer = @intCast(i), .release = release }, c, @intCast(i));

    const at: At = .{ .r = &r, .o = &m.owned, .release = release };
    const hidden: c_int = @intCast(c.hidden_size);
    var nb: [256]u8 = undefined;
    const embed = try r.tableAt(try at.name(&nb, "embed_tokens", "embed"), c.hidden_size);
    try m.owned.linear(embed);
    const head = try r.tableAt(if (release) try at.name(&nb, "", "unembed") else "lm_head", c.hidden_size);
    try m.owned.linear(head);
    var ones: [2]Array = undefined;
    for (&ones, [_]c_int{ hidden, @intCast(c.head_dim) }) |*x, n| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(x, &[_]c_int{n}, 1, .float32, r.s));
        try m.owned.own(x.*);
    }
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = hidden, .act = .bfloat16 },
        .embed_norm = try at.need("embed_norm.weight", "embed_norm.weight"),
        .final_norm = try at.need("norm.weight", "norm.weight"),
        .head = .{ .lin = head, .vocab = @intCast(c.unpadded_vocab_size), .softcap = c.logit_softcap },
        .body = .{
            .cfg = c.*,
            .layers = m.layers,
            .final_query = try at.query("final_input.query", "output_query", null, 1, hidden),
            .ones = ones[0],
            .ones_head = ones[1],
        },
    };
    if (c.vision) |v| m.tower = try vision.Tower.load(ctx, v, c.block_json);
    if (c.draft) |dc| if (draftsOn(device.deviceGeneration().gen, config.fusedOn())) if (try draft.Drafter.load(ctx, dc, c.hidden_size, &head, r.s)) |d| {
        m.drafter = d;
        m.tap_visits = dc.taps;
        m.spec = .{ .visits = m.tap_visits[0 .. dc.tap_count - 1], .keep = @intCast(dc.window - 1) };
        if (dc.fullBytesPerToken() > 0) m.spec.full = d;
        m.base.weights.spec = &m.spec;
    };
    // The parse's quant block is freed with it; the forward reads none of it.
    m.base.weights.body.cfg.block_json = "{}";

    return m;
}

/// Whether the bundled drafter loads: with the kernels on (only they serve a verify) and on an Apple9+ GPU (M3 on) or
/// an unknown one. On an M1 a few-row verify costs so much more than a step that drafted decode ran under serial.
fn draftsOn(gen: u32, fused: bool) bool {
    return fused and (gen == 0 or gen >= 15);
}

fn loadLayer(at: At, c: *const config.Config, i: u32) !layers.Layer {
    const hidden = c.hidden_size;
    const global = c.global[i];
    const in_norm: [3]Array = if (global)
        .{
            try at.need("q_input_layernorm.weight", "q_input_norm.weight"),
            try at.need("k_input_layernorm.weight", "k_input_norm.weight"),
            try at.need("v_input_layernorm.weight", "v_input_norm.weight"),
        }
    else
        @splat(try at.need("input_layernorm.weight", "attn_norm.weight"));
    var lins: [6]Linear = undefined;
    inline for (&lins, .{ "q_proj", "gate_proj", "r_proj", "sink_proj", "k_proj", "v_proj" }, .{ "wq_du", "gate_attn", "wr_du", "attn_sink_proj", "wk_dv", "wv_dv" }) |*l, leaf, rel|
        l.* = try at.projRaw("self_attn." ++ leaf, "attn." ++ rel, hidden);
    // Projections off one input join into one matmul; a sliding layer's k and v read the q input too.
    const quantized = lins[0].s.ctx != null;
    const join_q = if (quantized) try loader.join(at.o, &.{ &lins[0], &lins[1], &lins[2], &lins[3] }, at.r.s) else try at.joinRows(&.{ &lins[0], &lins[1], &lins[2], &lins[3] });
    const join_kv = if (global) b.linear.Joined{} else if (quantized) try loader.join(at.o, &.{ &lins[4], &lins[5] }, at.r.s) else try at.joinRows(&.{ &lins[4], &lins[5] });
    for (lins) |l| try at.o.linear(l);

    const rel_proj = try at.need("self_attn.rel_logits_proj.proj", "attn.rel_logits_proj.proj");
    var g = ops.Graph.init(gpa, at.r.s);
    defer g.deinit();
    const extent = ops.shape(rel_proj)[1];
    const rel_proj_rev = try at.keep(&g, try g.take(try g.astype(rel_proj, .float32), try g.arange(@floatFromInt(extent - 1), -1, -1, .int32), 1));
    var L: layers.Layer = .{
        .global = global,
        .boundary = i % c.block_every == 0,
        // Layer 0 has no attention input, so the release's query i - 1 is layer i's.
        .attn_query = if (i > 0) try at.query("attention_input.query", "attn_queries", i - 1, if (global) 3 else 1, @intCast(hidden)) else .{},
        .mlp_query = try at.query("mlp_input.query", "mlp_queries", i, 1, @intCast(hidden)),
        .in_norm = in_norm,
        .q = lins[0],
        .gate = lins[1],
        .r = lins[2],
        .sink = lins[3],
        .k = lins[4],
        .v = lins[5],
        .o = try at.proj("self_attn.o_proj", "attn.wo_ud", c.num_attention_heads * c.head_dim),
        .q_scale = try at.need("self_attn.q_l2_clip.scale", "attn.q_l2_clip.scale"),
        .rel_scale = try at.need("self_attn.rel_l2_clip.scale", "attn.rel_l2_clip.scale"),
        .rel_proj = rel_proj,
        .rel_proj_rev = rel_proj_rev,
        .k_conv = try at.conv("k", global, hidden),
        .v_conv = try at.conv("v", global, hidden),
        .attn_conv = try at.kernel("post_attention_convolution.conv1d.weight", "attn_sconv.weight"),
        .mlp_conv = try at.kernel("post_mlp_convolution.conv1d.weight", "mlp_sconv.weight"),
        .post_norm = try at.need("post_attention_layernorm.weight", "mlp_norm.weight"),
        .router = try at.float("mlp.gate.weight", "mlp.gate.weight", null),
        .router_bias = try at.float("mlp.gate.e_score_correction_bias", "mlp.gate.bias", null),
        .join_q = join_q,
        .join_kv = join_kv,
        .gate_up = undefined,
        .down = undefined,
    };
    L.gate_up = try at.bank("mlp.experts.gate_up_proj", "mlp.experts.w13", hidden, &L.gate_up_scale);
    L.down = try at.bank("mlp.experts.down_proj", "mlp.experts.w2", c.moe_intermediate_size, &L.down_scale);
    if (i >= c.core_start and i < c.core_end) {
        L.gates = try at.gates();
        L.message_query = try at.messageQuery(@intCast(c.core()), @intCast(hidden), @intCast(i - c.core_start));
    }
    return L;
}

/// The tiny pack the conformance suite runs, at `config.fixture_config`'s geometry: 8 layers of width 64 (global at
/// 1, 5 and 7), the loop over layers 2..6 three times, 4 experts of 24, 60 of 64 vocab rows real; seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = sdk.fixture;
    const V = 64;
    const H = 64;
    const KV = 32;
    const HD = 16;
    const HEADS = 4;
    const KVH = 2;
    const DR = 4;
    const E = 4;
    const MI = 24;
    const K = 4;
    const RANK = 8;
    const global = [_]bool{ false, true, false, false, false, true, false, true };
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    var seed: u64 = 43;
    const Put = struct {
        a: std.mem.Allocator,
        list: *std.ArrayList(fx.Tensor),
        seed: *u64,
        layer: ?usize = null,
        fn rand(p: @This(), comptime leaf: []const u8, shape: []const u64, scale: f32, offset: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            p.seed.* += 1;
            const data = try fx.randomF32(p.a, p.seed.*, n, scale);
            for (data) |*x| x.* += offset;
            const name = if (p.layer) |i|
                try std.fmt.allocPrint(p.a, "model.language_model.layers.{d}." ++ leaf, .{i})
            else if (comptime std.mem.startsWith(u8, leaf, "lm_head"))
                try p.a.dupe(u8, leaf)
            else
                try p.a.dupe(u8, "model.language_model." ++ leaf);
            try p.list.append(p.a, .{ .name = name, .shape = try p.a.dupe(u64, shape), .data = data });
        }
    };
    var p: Put = .{ .a = a, .list = &tensors, .seed = &seed };
    try p.rand("embed_tokens.weight", &.{ V, H }, 0.5, 0);
    try p.rand("lm_head.weight", &.{ V, H }, 0.5, 0);
    try p.rand("embed_norm.weight", &.{H}, 0.1, 1);
    try p.rand("norm.weight", &.{H}, 0.1, 1);
    try p.rand("final_input.query", &.{H}, 0.5, 0);
    try p.rand("message_input.query", &.{ 4, 4, H }, 0.5, 0);
    for (global, 0..) |g, i| {
        p.layer = i;
        if (i > 0) try p.rand("attention_input.query", if (g) &.{ 3, H } else &.{H}, 0.5, 0);
        try p.rand("mlp_input.query", &.{H}, 0.5, 0);
        if (g) {
            inline for (.{ "q", "k", "v" }) |x| try p.rand(x ++ "_input_layernorm.weight", &.{H}, 0.1, 1);
            inline for (.{ "k", "v" }) |x| {
                try p.rand("self_attn." ++ x ++ "_sconv.static_weight", &.{ K, KV }, 0.2, 0);
                try p.rand("self_attn." ++ x ++ "_sconv.weight_proj.weight", &.{ KVH * K, H }, 0.1, 0);
            }
        } else {
            try p.rand("input_layernorm.weight", &.{H}, 0.1, 1);
            inline for (.{ "k", "v" }) |x| try p.rand("self_attn." ++ x ++ "_sconv.conv1d.weight", &.{ KV, K, 1 }, 0.2, 0);
        }
        try p.rand("self_attn.q_proj.weight", &.{ HEADS * HD, H }, 0.2, 0);
        try p.rand("self_attn.k_proj.weight", &.{ KV, H }, 0.2, 0);
        try p.rand("self_attn.v_proj.weight", &.{ KV, H }, 0.2, 0);
        try p.rand("self_attn.r_proj.weight", &.{ HEADS * DR, H }, 0.2, 0);
        try p.rand("self_attn.o_proj.weight", &.{ H, HEADS * HD }, 0.1, 0);
        try p.rand("self_attn.gate_proj.weight", &.{ HEADS * HD, H }, 0.1, 0);
        try p.rand("self_attn.sink_proj.weight", &.{ HEADS, H }, 0.1, 0);
        try p.rand("self_attn.q_l2_clip.scale", &.{ HEADS, HD }, 0.1, 1);
        try p.rand("self_attn.rel_l2_clip.scale", &.{ HEADS, DR }, 0.1, 1);
        try p.rand("self_attn.rel_logits_proj.proj", &.{ DR, if (g) 8 else 6 }, 0.5, 0);
        try p.rand("post_attention_convolution.conv1d.weight", &.{ H, K, 1 }, 0.2, 0);
        try p.rand("post_mlp_convolution.conv1d.weight", &.{ H, K, 1 }, 0.2, 0);
        try p.rand("post_attention_layernorm.weight", &.{H}, 0.1, 1);
        try p.rand("mlp.gate.weight", &.{ E, H }, 0.2, 0);
        try p.rand("mlp.gate.e_score_correction_bias", &.{E}, 0.1, 0);
        try p.rand("mlp.experts.gate_up_proj.weight", &.{ E, 2 * MI, H }, 0.1, 0);
        try p.rand("mlp.experts.down_proj.weight", &.{ E, H, MI }, 0.1, 0);
        if (i >= 2 and i < 6) inline for (.{ "q", "k", "v", "mlp" }) |x| {
            try p.rand(x ++ "_message_gate.down_proj.weight", &.{ RANK, 2 * H }, 0.1, 0);
            try p.rand(x ++ "_message_gate.forget_gate_proj.weight", &.{ H, RANK }, 0.3, 0);
            try p.rand(x ++ "_message_gate.message_gate_proj.weight", &.{ H, RANK }, 0.3, 0);
        };
    }
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes = config.fixture_config },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
    _ = layers;
    _ = vision;
    _ = media;
    _ = draft;
}

// Bar: two slots that take turns each keep the taps of their own sequence; with every place taken a new slot
// replaces the ring parked longest.
test "inkling2 draft taps: each engine slot keeps its own ring" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var sp: Spec = .{ .visits = &.{}, .keep = 4 };
    defer sp.deinit();
    const row = try g.ones(&.{ 1, 1, 2 }, .float32);
    sp.select(7);
    try sp.push(&g, 10, row);
    try sp.push(&g, 11, row);
    sp.select(9);
    try std.testing.expectEqual(@as(c_int, 0), sp.rows(12));
    try sp.push(&g, 12, row);
    sp.select(7);
    try std.testing.expectEqual(@as(c_int, 2), sp.rows(12));
    sp.select(9);
    try std.testing.expectEqual(@as(c_int, 1), sp.rows(13));
    // Slot 1 parks 9 and slot 2 parks 1 in the free places; slot 3 then takes the place of 7, parked longest.
    for ([_]u32{ 1, 2, 3 }) |id| {
        sp.select(id);
        try sp.push(&g, 0, row);
    }
    sp.select(9);
    try std.testing.expectEqual(@as(c_int, 1), sp.rows(13));
    sp.select(7);
    try std.testing.expectEqual(@as(c_int, 0), sp.rows(12));
}

// Bar: the drafter loads from an M3 on (and where the generation is unknown), never on an M1 (g13) or M2 (g14).
test "inkling2: the bundled drafter loads on Apple9 and later only" {
    for ([_]u32{ 13, 14 }) |g| try std.testing.expect(!draftsOn(g, true));
    for ([_]u32{ 0, 15, 16, 17 }) |g| try std.testing.expect(draftsOn(g, true));
    try std.testing.expect(!draftsOn(17, false));
}

// Bar: a round verifies drafts while their running probability clears the cut, never a stop: the trunk's token carries it.
test "inkling2 draft: a round never verifies a stop" {
    const sure = [_]f32{ 0.99, 0.99, 0.99, 0.99 };
    const tokens = [_]i32{ 5, 9, 2, 8 };
    try std.testing.expectEqual(@as(u32, 4), draftsToVerify(&sure, &tokens, 0.5, &.{}));
    try std.testing.expectEqual(@as(u32, 1), draftsToVerify(&sure, &tokens, 0.5, &.{ 70, 9 }));
    try std.testing.expectEqual(@as(u32, 0), draftsToVerify(&sure, &tokens, 0.5, &.{5}));
    try std.testing.expectEqual(@as(u32, 2), draftsToVerify(&.{ 0.9, 0.6, 0.8, 0.9 }, &tokens, 0.5, &.{}));
}
