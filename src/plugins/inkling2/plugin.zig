//! Thinking Machines Inkling 2 Mini (`inkling2_loop`, text and images), in mlx-lm's layout (`models/inkling2`) or as released:
//! 24 layers of weights applied 48 times per token, the middle 12 run three times, each pass with its own KV cache and conv
//! windows and, after the first, messages from the pass before. The stream between layers is depth attention over
//! saved blocks, not a plain residual, so the forward is the plugin's own loop (`layers.zig`), not a base's.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const env = @import("../../env.zig");
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
const dflash2 = @import("dflash2.zig");
const kv = @import("../../sdk/bases/kv.zig");
const rounds = @import("../../sdk/spec/rounds.zig");
const sdk_host = @import("../../sdk/host.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const gpa = std.heap.c_allocator;

/// MLX's nvfp4 global scale is the block's amax; the codes times their block scales carry it over 6 x 448.
const nvfp4_global_unit: f32 = 6.0 * 448.0;

/// The slots whose rounds one verify carries, and whose rings the Spec keeps (the current one and `max_open - 1` parked).
const max_open = 16;

/// Online per-slot logistic calibration for DFlash2 tree node selection (`logit(p) = slope * logit(q) + intercept[pos]`).
/// Kept per slot on `Spec` so fresh greedy requests start from the identical deterministic prior.
const TreeCalib = struct {
    slope: f32 = 0.7,
    intercepts: [8]f32 = .{ 0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45, -0.45 },

    fn observe(c: *TreeCalib, pos: usize, q: f32, accepted: bool) void {
        if (pos >= c.intercepts.len or !(q > 0.0) or !(q < 1.0)) return;
        const qc = std.math.clamp(q, 1e-6, 1.0 - 1e-6);
        const lq = @log(qc / (1.0 - qc));
        const z = c.slope * lq + c.intercepts[pos];
        const p = 1.0 / (1.0 + @exp(-z));
        const err = @as(f32, if (accepted) 1.0 else 0.0) - p;
        const lr: f32 = 0.04;
        c.intercepts[pos] = std.math.clamp(c.intercepts[pos] + lr * err, -3.5, 2.5);
        c.slope = std.math.clamp(c.slope + (lr * 0.25) * err * lq, 0.25, 2.0);
    }
};

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
    /// Positions of an adopted ring (`adopt`) the new request's sequence shares; the next push may cut back to them.
    agree: u64 = 0,
    /// What the drafter's confidence in a draft has been worth on the slot's sequence (another's says nothing of it).
    calib: rounds.Calibration = .{ .head = true },
    /// Per-slot online logistic calibration for DFlash2 tree expansion.
    tree_calib: TreeCalib = .{},
    /// Decayed acceptance counts for sampled bundled-drafter group sizing (`stepGroupBundled`).
    group_hit: f32 = 0,
    group_seen: f32 = 0,
    parked: [max_open - 1]Parked = @splat(.{}),
    clock: u64 = 0,
    /// Set for a verify: the visits read the caches without appending and leave their rows here.
    opens: ?[]kernels.Open = null,
    /// A verify's taps `[1, S, taps * hidden]`, held until its accept.
    held: Array = .{},
    /// Set when a drafter layer attends every committed position: a push hands it each forward's rows, all of them.
    full: ?*const draft.Drafter = null,

    const Parked = struct { owner: u32 = 0, ring: Array = .{}, end: u64 = 0, at: u64 = 0, context: draft.ContextKv = .{}, calib: rounds.Calibration = .{ .head = true }, tree_calib: TreeCalib = .{}, group_hit: f32 = 0, group_seen: f32 = 0 };

    /// Makes slot `from`'s ring slot `slot`'s, valid for the positions before `agree` (`Forward.adopt_slot`): a new
    /// request that repeats `from`'s sequence that far continues its drafter.
    pub fn adopt(sp: *Spec, from: u32, slot: u32, agree: u64) void {
        sp.select(from);
        sp.owner = slot;
        sp.agree = agree;
    }

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
        sp.parked[to] = .{ .owner = sp.owner, .ring = sp.ring, .end = sp.end, .at = sp.clock, .context = sp.context, .calib = sp.calib, .tree_calib = sp.tree_calib, .group_hit = sp.group_hit, .group_seen = sp.group_seen };
        sp.ring = if (found) theirs.ring else .{};
        sp.end = if (found) theirs.end else 0;
        sp.context = if (found) theirs.context else .{};
        sp.calib = if (found) theirs.calib else .{ .head = true };
        sp.tree_calib = if (found) theirs.tree_calib else .{};
        sp.group_hit = if (found) theirs.group_hit else 0;
        sp.group_seen = if (found) theirs.group_seen else 0;
        sp.owner = slot;
    }

    /// The context rows before `position`; 0 when the ring ends elsewhere.
    pub fn rows(sp: *const Spec, position: u64) c_int {
        return if (sp.ring.ctx != null and sp.end == position) ops.shape(sp.ring)[1] else 0;
    }

    /// The taps of the rows from `first` on (`parts` side by side): they extend the ring or restart it.
    fn push(sp: *Spec, g: *ops.Graph, first: u64, parts: []const ops.Array) !void {
        try sp.pushOpt(g, first, parts, true);
    }

    fn pushOpt(sp: *Spec, g: *ops.Graph, first: u64, parts: []const ops.Array, flush: bool) !void {
        const n = ops.shape(parts[0])[1];
        // An adopted ring the request repeats past `first` is cut back to it, its newest rows a later turn's.
        if (sp.ring.ctx != null and first < sp.end and first <= sp.agree and sp.end - first < ops.shape(sp.ring)[1]) {
            const cut = try g.contiguous(try g.sliceAxis(sp.ring, 1, 0, ops.shape(sp.ring)[1] - @as(c_int, @intCast(sp.end - first))));
            _ = mlx.mlx_array_free(sp.ring);
            sp.ring = g.escape(cut);
            sp.end = first;
            sp.context.cut(first);
        }
        sp.agree = 0;
        const continues = sp.rows(first) > 0;
        if (!continues) sp.context.release();
        if (sp.full) |d| try d.appendFull(g, &sp.context, parts, first);
        const taps = try draft.Drafter.sideBySide(g, parts, @max(0, n - sp.keep), n);
        var all = if (continues) try g.concat(&.{ sp.ring, taps }, 1) else taps;
        const total = ops.shape(all)[1];
        if (total > sp.keep) all = try g.contiguous(try g.sliceAxis(all, 1, total - sp.keep, total));
        const next = g.escape(all);
        // Evaluated now: left lazy, each forward would hang another concat off the last.
        if (flush) try ops.asyncMany(&.{next});
        if (sp.ring.ctx != null) _ = mlx.mlx_array_free(sp.ring);
        sp.ring = next;
        sp.end = first + @as(u64, @intCast(n));
    }

    fn deinit(sp: *Spec) void {
        for ([_]Array{ sp.ring, sp.held }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (sp.parked) |p| if (p.ring.ctx != null) {
            _ = mlx.mlx_array_free(p.ring);
        };
        sp.context.release();
        for (&sp.parked) |*p| p.context.release();
    }
};

pub const Model = struct {
    pub const batches_decode = true;
    pub const prompt_last_row = true;

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
        // A verify of several slots' blocks: all their rows as one sequence, each slot's behind its own cache and state.
        if (run.rows > 1 and w.spec != null and w.spec.?.opens != null) return sequence(w, shared, g, try g.reshape(h, &.{ 1, run.slots * run.rows, -1 }), run);
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
        const grouped = run.slots > 1 and run.rows > 1;
        const batched = run.slots > 1 and !grouped;

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
            const from = if (sp.full != null or grouped) 0 else @max(0, run.rows - sp.keep);
            const end = if (grouped) run.rows * run.slots else run.rows;
            var parts: [config.max_taps]ops.Array = undefined;
            for (taps[0 .. sp.visits.len + 1], parts[0 .. sp.visits.len + 1]) |x, *part| part.* = try g.sliceAxis(x, 1, from, end);
            if (sp.opens != null) {
                if (sp.held.ctx != null) _ = mlx.mlx_array_free(sp.held);
                sp.held = g.escape(try g.concat(parts[0 .. sp.visits.len + 1], 2));
            } else try sp.push(g, run.positions[0] + @as(u64, @intCast(from)), parts[0 .. sp.visits.len + 1]);
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
    .spec = &.{ .native_drafter = &.{ .step = &draftStep, .verify = &draftVerify, .finish = &draftFinish, .active = &draftActive, .verify_group = &draftVerifyGroup, .finish_group = &draftFinishGroup, .group_width = &draftGroupWidth, .step_group = &draftStepGroup } },
};

fn draftStep(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    round(m, f, out) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

fn draftStepGroup(mp: *anyopaque, fs: [*]const abi.Forward, outs: [*]abi.ForwardOut, n: u32) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (n == 0 or n > max_open) return .{ .code = .failed, .why = abi.Str.of("Inkling2TooManyOpen") };
    return stepGroup(m, fs[0..n], outs[0..n]) catch |e| .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
}

fn draftVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    verifyGroup(m, solo(f), @as([*]abi.ForwardOut, @ptrCast(out))[0..1]) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

fn draftVerifyGroup(mp: *anyopaque, fs: [*]const abi.Forward, outs: [*]abi.ForwardOut, n: u32) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (n == 0 or n > max_open) return .{ .code = .failed, .why = abi.Str.of("Inkling2TooManyOpen") };
    verifyGroup(m, fs[0..n], outs[0..n]) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

/// Slots whose rounds pay as one verify. The DFlash2 draft's: `max_open` on the tensor unit (it reads a block's keys for
/// four slots a launch), else 8. The bundled head's: 8 on the tensor unit (batched draft + BM=8 grouped MoE), else 2.
pub fn groupWidth(dflash2_draft: bool, nax: bool) u32 {
    if (!dflash2_draft) return if (nax) 8 else 2;
    return if (nax) max_open else 8;
}

fn draftGroupWidth(mp: *anyopaque) callconv(.c) u32 {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return groupWidth(m.drafter2 != null, device.naxAvailable());
}

fn draftActive(mp: *anyopaque) callconv(.c) bool {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return m.blockWidth() > 0;
}

fn draftFinish(mp: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    finishGroup(m, null, &.{accepted}, @as([*]abi.ForwardOut, @ptrCast(out))[0..1]) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

fn draftFinishGroup(mp: *anyopaque, slots: [*]const u32, accepted: [*]const u32, outs: [*]abi.ForwardOut, n: u32) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (n == 0 or n > max_open) return .{ .code = .failed, .why = abi.Str.of("Inkling2TooManyOpen") };
    finishGroup(m, slots[0..n], accepted[0..n], outs[0..n]) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    return .ok;
}

/// The trunk's pick per row of `logits` `[rows, vocab]` under the sampler's suppress mask `[vocab]` bool: `[rows]`.
fn picks(g: *ops.Graph, logits: ops.Array, suppress: abi.Array) !ops.Array {
    const masked = if (suppress.ctx == null) logits else try g.where(.{ .ctx = suppress.ctx }, try g.scalar(-std.math.inf(f32), .float32), logits);
    return g.astype(try g.argmax(masked, -1, false), .int32);
}

/// One slot's share of an open verify (or of its plain step): the drafter proposed after the committed token `t1`.
const Lane = struct {
    slot: u32 = 0,
    position: u32 = 0,
    cache: ?abi.Handle = null,
    drafts: u32 = 0,
    /// `t1`, then the drafts.
    ids: [abi.max_round_tokens]u32 = undefined,
    /// The law sampled drafts were drawn from `[drafts, vocab]` f32 (`Forward.draft_temperature`); unset for point masses.
    law: ops.Array = .{},
    /// The drafter's confidence in each argmax draft it verifies (none for a copy or a drawn draft).
    sure: [abi.max_round_tokens]f32 = undefined,
    sure_n: u32 = 0,
    /// A sampled block's ids, still on the GPU behind the verify's graph: read by `settleDraft`.
    pending: ops.Array = .{},
    /// `t1`'s ids `[1]`, the drafts proposed after it, and what the lane asked of the drafter (`draftWanted`).
    first: ops.Array = .{},
    proposed: ops.Array = .{},
    want: ?Want = null,
};

/// A lane's call on the drafter: `t1`'s normed embedding, the drafts it may make, the law to draw them from.
const Want = struct { anchor: ops.Array, cap: u32, law: ?draft.Drafter.Law, round_ms: []const f32, stops: []const u32 };

/// A round's open verify (or its plain step) of one slot or several: the trunk read each slot's `t1` and the drafts the
/// drafter was sure of as ONE forward that appends nothing, `rows` rows a slot (the shorter blocks padded), its graph
/// keeping what a commit reads (each visit's rows in `opens`, the taps in `held`). One slot with no context, or no draft
/// worth a row, took a plain step that appended `t1` already.
const Opened = struct {
    g: *ops.Graph,
    fo: abi.ForwardOut = .{},
    opens: []kernels.Open = &.{},
    cache_ops: ?*const abi.CacheOps = null,
    lanes: [max_open]Lane = undefined,
    n: usize = 0,
    rows: u32 = 1,
    held: ops.Array = .{},

    fn deinit(o: *Opened) void {
        for ([_]abi.Array{ o.fo.logits, o.fo.hidden, .{ .ctx = o.held.ctx } }) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
        };
        gpa.free(o.opens);
        o.g.deinit();
        gpa.destroy(o.g);
    }

    fn plain(o: *const Opened) bool {
        return o.opens.len == 0;
    }
};

/// A round's forward goes to the GPU every this many visits while the rest of its graph is built.
const pipeline_layers: u32 = 4;

fn stopIds(f: *const abi.Forward) []const u32 {
    return if (f.size >= @offsetOf(abi.Forward, "stop_ids") + @sizeOf(abi.Span(u32))) f.stop_ids.slice() else &.{};
}

/// `lane`'s first row and the drafts the engine gave it (a copy of its context); a lane the drafter should draft for is
/// left asking (`Lane.want`, answered by `draftWanted`).
fn draftLane(m: *Loaded, o: *Opened, lane: *Lane, f: *const abi.Forward) !void {
    const w = &m.base.weights;
    const cfg = &w.body.cfg;
    const g = o.g;
    const t1 = f.tokens.slice()[0];
    const position = lane.position;
    lane.ids[0] = t1;
    const sp = w.spec;
    if (sp) |x| x.select(lane.slot);
    const context: c_int = if (sp) |x| x.rows(position) else 0;
    const cap: u32 = if (m.blockWidth() > 0) @min(f.accept_cap, m.blockWidth() - 1) else 0;
    lane.first = try g.fromSlice(i32, &.{@intCast(t1)}, &.{1});
    const given: []const u32 = if (f.size >= @offsetOf(abi.Forward, "draft_ids") + @sizeOf(abi.Span(u32))) f.draft_ids.slice() else &.{};
    // A sampled round's width is the engine's (`accept_cap`), chosen before any draft is drawn: the drafter makes
    // only that many.
    const law: ?draft.Drafter.Law = if (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64) and f.draft_temperature > 0)
        .{ .temperature = f.draft_temperature, .seed = f.draft_seed, .steps = cap }
    else
        null;
    const round_ms: []const f32 = if (f.size >= @offsetOf(abi.Forward, "round_ms") + @sizeOf(abi.Span(f32))) f.round_ms.slice() else &.{};
    const stops = stopIds(f);
    if (given.len > 0) {
        // The engine's drafts (a copy of the context): verified as given, up to a stop id, which ends the turn.
        var ids: [abi.max_round_tokens]i32 = undefined;
        for (given[0..@min(given.len, f.accept_cap, abi.max_round_tokens - 1)]) |id| {
            if (std.mem.indexOfScalar(u32, stops, id) != null) break;
            ids[lane.drafts] = @intCast(id);
            lane.ids[1 + lane.drafts] = id;
            lane.drafts += 1;
        }
        if (lane.drafts > 0) lane.proposed = try g.fromSlice(i32, ids[0..lane.drafts], &.{@intCast(lane.drafts)});
    } else if (cap > 0 and context > 0) {
        const pos = [_]u32{position};
        const r: run_mod.Run = .{ .mode = .verify, .rows = 1, .positions = &pos, .caches = f.caches.slice(), .cache_ops = f.cache_ops };
        var c = r.ctx(g, &m.base.shared);
        const anchor = try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &c, try g.reshape(lane.first, &.{ 1, 1 })), w.embed_norm, cfg.rms_norm_eps);
        lane.want = .{ .anchor = anchor, .cap = cap, .law = law, .round_ms = round_ms, .stops = stops };
    }
}

/// Answers every lane's call on the drafter: multi-lane calls batch through `dflash2.draftBatch` or `draft.draftBatch`,
/// sharing layer projections, `lm_head` and the Markov fold read across lanes.
fn draftWanted(m: *Loaded, o: *Opened) !void {
    const w = &m.base.weights;
    const sp = w.spec.?;
    const g = o.g;
    var asking: [max_open]*Lane = undefined;
    var n: usize = 0;
    for (o.lanes[0..o.n]) |*lane| if (lane.want != null) {
        asking[n] = lane;
        n += 1;
    };
    if (n == 0) return;
    if (m.drafter2) |d2| {
        var batched = n > 1;
        for (asking[0..n]) |lane| batched = batched and lane.want.?.law != null;
        if (batched) {
            var contexts: [max_open]draft.ContextKv = undefined;
            var lanes: [max_open]dflash2.Drafter.DraftLane = undefined;
            var temperatures: [max_open]f32 = undefined;
            for (asking[0..n], 0..) |lane, k| {
                sp.select(lane.slot);
                const want = lane.want.?;
                contexts[k] = sp.context;
                lanes[k] = .{ .t1 = lane.first, .anchor = want.anchor, .taps = sp.ring, .position = @intCast(lane.position), .cache = &contexts[k], .steps = want.cap };
                temperatures[k] = want.law.?.temperature;
            }
            var blocks: [max_open]draft.Drafter.Block = undefined;
            try d2.draftBatch(g, lanes[0..n], .{ .temperatures = temperatures[0..n], .seed = asking[0].want.?.law.?.seed }, &w.head.lin, w.head.vocab, &blocks);
            for (asking[0..n], blocks[0..n], 0..) |lane, drafted, k| {
                sp.select(lane.slot);
                sp.context = contexts[k];
                lane.proposed = try adoptDrafted(g, sp, lane, drafted);
            }
            return;
        }
    } else if (m.drafter) |d1| if (n > 1) {
        var all_sampled = true;
        var all_greedy = true;
        for (asking[0..n]) |lane| {
            all_sampled = all_sampled and lane.want.?.law != null;
            all_greedy = all_greedy and lane.want.?.law == null;
        }
        if (all_sampled or all_greedy) {
            var contexts: [max_open]draft.ContextKv = undefined;
            var lanes: [max_open]draft.Drafter.DraftLane = undefined;
            var temperatures: [max_open]f32 = undefined;
            for (asking[0..n], 0..) |lane, k| {
                sp.select(lane.slot);
                const want = lane.want.?;
                contexts[k] = sp.context;
                lanes[k] = .{ .t1 = lane.first, .anchor = want.anchor, .taps = sp.ring, .position = @intCast(lane.position), .cache = &contexts[k], .steps = want.cap };
                if (all_sampled) temperatures[k] = want.law.?.temperature;
            }
            if (all_sampled) {
                var blocks: [max_open]draft.Drafter.Block = undefined;
                try d1.draftBatch(g, lanes[0..n], &w.head.lin, w.head.vocab, w.head.softcap, .{ .temperatures = temperatures[0..n], .seed = asking[0].want.?.law.?.seed }, &blocks);
                for (asking[0..n], blocks[0..n], 0..) |lane, drafted, k| {
                    sp.select(lane.slot);
                    sp.context = contexts[k];
                    lane.proposed = try adoptDrafted(g, sp, lane, drafted);
                }
            } else {
                const packed_b = try d1.draftBatchPacked(g, lanes[0..n], &w.head.lin, w.head.vocab, w.head.softcap, null);
                try ops.evalMany(&.{ packed_b.tokens, packed_b.confidence });
                const cols: usize = @intCast(ops.shape(packed_b.tokens)[1]);
                const all_toks = mlx.mlx_array_data_int32(packed_b.tokens) orelse return error.Inkling2DraftRound;
                const all_sure = mlx.mlx_array_data_float32(packed_b.confidence) orelse return error.Inkling2DraftRound;
                for (asking[0..n], 0..) |lane, k| {
                    sp.select(lane.slot);
                    sp.context = contexts[k];
                    const want = lane.want.?;
                    const row_toks = all_toks[k * cols ..][0..cols];
                    const row_sure = all_sure[k * cols ..][0..cols];
                    const cap = @min(want.cap, @as(u32, @intCast(cols)));
                    lane.drafts = draftsToVerify(row_sure[0..cap], row_toks[0..cap], want.stops, &sp.calib, want.round_ms);
                    for (0..lane.drafts) |j| lane.ids[1 + j] = @intCast(row_toks[j]);
                    lane.sure_n = lane.drafts;
                    @memcpy(lane.sure[0..lane.drafts], row_sure[0..lane.drafts]);
                    lane.proposed = if (lane.drafts > 0) try g.fromSlice(i32, row_toks[0..lane.drafts], &.{@intCast(lane.drafts)}) else .{};
                }
            }
            return;
        }
    };
    for (asking[0..n]) |lane| {
        sp.select(lane.slot);
        const want = lane.want.?;
        const drafted = if (m.drafter2) |d2|
            try d2.draft(g, lane.first, want.anchor, sp.ring, @intCast(lane.position), &w.head.lin, w.head.vocab, &sp.context, want.law)
        else
            try m.drafter.?.draft(g, lane.first, want.anchor, sp.ring, @intCast(lane.position), &w.head.lin, w.head.vocab, w.head.softcap, &sp.context, want.law);
        lane.proposed = try adoptDrafted(g, sp, lane, drafted);
    }
}

/// The ids `lane` verifies after `t1` from the block the drafter made. A sampled block (`law`) starts on the GPU while the
/// verify's graph is built behind it, its ids read by `settleDraft`; an argmax one is read now and cut where the drafter's
/// confidence stops paying (`draftsToVerify`).
fn adoptDrafted(g: *ops.Graph, sp: *Spec, lane: *Lane, drafted: draft.Drafter.Block) !ops.Array {
    const want = lane.want.?;
    if (want.law != null) {
        const started = [_]ops.Array{ drafted.tokens, drafted.probs };
        const vec = mlx.mlx_vector_array_new_data(&started, started.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
        lane.drafts = want.cap;
        lane.pending = drafted.tokens;
        lane.law = drafted.probs;
        return drafted.tokens;
    }
    try ops.evalMany(&.{ drafted.tokens, drafted.confidence });
    const sure = mlx.mlx_array_data_float32(drafted.confidence) orelse return error.Inkling2DraftRound;
    const tokens = mlx.mlx_array_data_int32(drafted.tokens) orelse return error.Inkling2DraftRound;
    lane.drafts = draftsToVerify(sure[0..want.cap], tokens[0..want.cap], want.stops, &sp.calib, want.round_ms);
    for (0..lane.drafts) |k| lane.ids[1 + k] = @intCast(tokens[k]);
    lane.sure_n = lane.drafts;
    @memcpy(lane.sure[0..lane.drafts], sure[0..lane.drafts]);
    return g.sliceAxis(drafted.tokens, 0, 0, @intCast(lane.drafts));
}

/// Drafts after each of `fs`' tokens (a slot each) and reads their blocks as one forward (or steps plainly): `fo.logits`
/// are the rows', `rows` a slot.
fn openGroup(m: *Loaded, fs: []const abi.Forward) !*Opened {
    const w = &m.base.weights;
    const cfg = &w.body.cfg;
    const g = try gpa.create(ops.Graph);
    g.* = ops.Graph.init(gpa, m.base.s);
    const o = try gpa.create(Opened);
    o.* = .{ .g = g, .cache_ops = fs[0].cache_ops };
    errdefer {
        o.deinit();
        gpa.destroy(o);
    }
    var ids: [max_open]ops.Array = undefined;
    var slot_ids: [max_open]u32 = undefined;
    var positions: [max_open]u32 = undefined;
    var handles: [max_open]abi.Handle = undefined;
    for (fs, 0..) |*f, i| {
        const lane = &o.lanes[i];
        lane.* = .{ .slot = f.slots.slice()[0], .position = f.positions.slice()[0], .cache = f.caches.slice()[0] };
        slot_ids[i] = lane.slot;
        positions[i] = lane.position;
        handles[i] = lane.cache.?;
        o.n = i + 1;
        try draftLane(m, o, lane, f);
    }
    try draftWanted(m, o);
    for (o.lanes[0..o.n], ids[0..o.n]) |*lane, *x| x.* = if (lane.drafts == 0) lane.first else try g.concat(&.{ lane.first, lane.proposed }, 0);
    if (o.n == 1 and o.lanes[0].drafts == 0) {
        const f = &fs[0];
        m.split.follow(true);
        const v = run_mod.tableForward(Model, &m.base, &.{ .mode = .decode, .logits = .last_row, .rows = 1, .tokens = f.tokens, .slots = f.slots, .positions = f.positions, .caches = f.caches, .cache_ops = f.cache_ops, .pipeline_every = pipeline_layers }, &o.fo);
        if (!v.isOk()) return error.Inkling2DraftRound;
        return o;
    }
    var rows: u32 = 2;
    for (o.lanes[0..o.n]) |lane| rows = @max(rows, 1 + lane.drafts);
    o.rows = rows;
    var parts: [2 * max_open]ops.Array = undefined;
    var k: usize = 0;
    for (o.lanes[0..o.n], ids[0..o.n]) |lane, lane_ids| {
        parts[k] = lane_ids;
        k += 1;
        if (rows > 1 + lane.drafts) {
            parts[k] = try g.zeros(&.{@intCast(rows - 1 - lane.drafts)}, .int32);
            k += 1;
        }
    }
    const sp = w.spec.?;
    o.opens = try gpa.alloc(kernels.Open, cfg.visits());
    @memset(o.opens, .{ .g = g });
    sp.opens = o.opens;
    defer sp.opens = null;
    const all_ids = try g.concat(parts[0..k], 0);
    m.split.follow(false);
    const v = run_mod.tableForward(Model, &m.base, &.{
        .mode = if (o.n == 1) .verify else .batched_rows,
        .logits = .all_rows,
        .rows = @intCast(rows),
        .tokens = .{},
        .ids = .{ .ctx = all_ids.ctx },
        .slots = .of(slot_ids[0..o.n]),
        .positions = .of(positions[0..o.n]),
        .caches = .of(handles[0..o.n]),
        .cache_ops = o.cache_ops,
        .pipeline_every = pipeline_layers,
    }, &o.fo);
    if (!v.isOk()) return error.Inkling2DraftRound;
    // Other slots' rounds may open before this one's accept: its taps stay with it.
    o.held = sp.held;
    sp.held = .{};
    return o;
}

/// The drafts a round verifies. An argmax block stops before a stop id (`Forward.stop_ids`), which ends the turn, and
/// keeps the count that buys the most expected tokens per ms of the round (`round_ms` by the drafts verified,
/// `Forward.round_ms`; empty = every draft), each draft's acceptance read off the drafter's confidence in it. A drawn
/// block (`calib` null) is as wide as it was asked and ends on its stop id, which the engine tests and never commits:
/// a cut on what was drawn skews the output, and the turn ended less often than the trunk ends it.
fn draftsToVerify(sure: []const f32, tokens: []const i32, stops: []const u32, calib: ?*const rounds.Calibration, round_ms: []const f32) u32 {
    var n: u32 = 0;
    while (n < sure.len and n < rounds.MAX_DRAFTS and std.mem.indexOfScalar(u32, stops, @intCast(tokens[n])) == null) n += 1;
    const cal = calib orelse return @intCast(@min(n + 1, sure.len, rounds.MAX_DRAFTS));
    if (round_ms.len < n + 1) return n;
    var p: [rounds.MAX_DRAFTS]f32 = undefined;
    for (sure[0..n], p[0..n]) |c, *o| o.* = cal.probability(c);
    var costs: rounds.Costs = @splat(std.math.inf(f32));
    @memcpy(costs[1 .. n + 2], round_ms[0 .. n + 1]);
    return rounds.chooseRows(p[0..n], &costs).rows - 1;
}

/// Keeps `t1` and the first `accepted` drafts of lane `i`: their keys, values and conv windows committed, as the serial
/// steps would have left them, and their taps on the ring. The position after them.
fn commitLane(m: *Loaded, o: *Opened, i: usize, accepted: u32) !u64 {
    const lane = &o.lanes[i];
    if (o.plain()) return @as(u64, lane.position) + 1;
    if (accepted > lane.drafts) return error.Inkling2DraftRound;
    const sp = m.base.weights.spec.?;
    sp.select(lane.slot);
    for (lane.sure[0..@min(lane.sure_n, accepted + 1)], 0..) |c, k| sp.calib.observe(c, k < accepted);
    const kept: c_int = @intCast(1 + accepted);
    const handles = [_]abi.Handle{lane.cache orelse return error.Inkling2DraftRound};
    const positions = [_]u32{lane.position};
    const r: run_mod.Run = .{ .mode = .verify, .rows = kept, .positions = &positions, .caches = &handles, .cache_ops = o.cache_ops };
    var c = r.ctx(o.g, &m.base.shared);
    try kernels.commit(&c, &m.base.weights.body.cfg, o.opens, kept, i, o.n);
    const at: c_int = @intCast(i * o.rows);
    try sp.push(o.g, lane.position, &.{try o.g.sliceAxis(o.held, 1, at, at + kept)});
    return @as(u64, lane.position) + @as(u64, @intCast(kept));
}

fn solo(f: *const abi.Forward) []const abi.Forward {
    return @as([*]const abi.Forward, @ptrCast(f))[0..1];
}

fn closeOpen(o: *Opened) void {
    o.deinit();
    gpa.destroy(o);
}

/// One greedy round (`NativeDrafter.step`).
fn round(m: *Loaded, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const o = try openGroup(m, solo(f));
    defer closeOpen(o);
    const lane = &o.lanes[0];
    const picked = try picks(o.g, .{ .ctx = o.fo.logits.ctx }, f.suppress_mask);
    try ops.eval(picked);
    const want = mlx.mlx_array_data_int32(picked) orelse return error.Inkling2DraftRound;
    var accepted: u32 = 0;
    while (accepted < lane.drafts and want[accepted] == @as(i32, @intCast(lane.ids[1 + accepted]))) accepted += 1;
    const position = try commitLane(m, o, 0, accepted);
    out.round = .{ .n = 1 + accepted, .accepted = accepted, .next_token = @intCast(want[accepted]), .rows = 1 + lane.drafts, .position = position };
    @memcpy(out.round.tokens[0 .. 1 + accepted], lane.ids[0 .. 1 + accepted]);
}

/// The first half of a sampled round (`NativeDrafter.verify`, `verify_group`): each request's rows' ids and logits
/// `[rows, vocab]` f32 for the engine to accept, the verify kept open until `finishGroup` (a new one discards it).
fn verifyGroup(m: *Loaded, fs: []const abi.Forward, outs: []abi.ForwardOut) !void {
    if (m.opened) |x| {
        closeOpen(x);
        m.opened = null;
    }
    const o = try openGroup(m, fs);
    errdefer closeOpen(o);
    const rows: c_int = @intCast(o.rows);
    // Left lazy: the engine's accept evaluates the verify with it, one evaluation a round.
    const all = try o.g.astype(try o.g.reshape(.{ .ctx = o.fo.logits.ctx }, &.{ rows * @as(c_int, @intCast(o.n)), -1 }), .float32);
    for (fs, outs, o.lanes[0..o.n], 0..) |*f, *out, *lane, i| {
        if (lane.pending.ctx != null) try settleDraft(o.g, lane, stopIds(f));
        const n: u32 = 1 + lane.drafts;
        const at: c_int = @intCast(i * o.rows);
        out.logits = .{ .ctx = o.g.escape(if (o.n == 1 and n == o.rows) all else try o.g.sliceAxis(all, 0, at, at + @as(c_int, @intCast(n)))).ctx };
        if (lane.law.ctx != null and out.size >= @offsetOf(abi.ForwardOut, "draft_probs") + @sizeOf(abi.Array)) out.draft_probs = .{ .ctx = o.g.escape(lane.law).ctx };
        out.round = .{ .n = n, .rows = n, .position = lane.position };
        @memcpy(out.round.tokens[0..n], lane.ids[0..n]);
    }
    m.opened = o;
}

/// Reads the ids of a sampled draft left on the GPU (`Lane.pending`), now that the verify's graph stands, and ends the
/// block at the first stop id (which the engine tests and never commits): the verify's later rows go unread.
fn settleDraft(g: *ops.Graph, lane: *Lane, stops: []const u32) !void {
    try ops.eval(lane.pending);
    const tokens = mlx.mlx_array_data_int32(lane.pending) orelse return error.Inkling2DraftRound;
    var n: u32 = 0;
    while (n < lane.drafts and std.mem.indexOfScalar(u32, stops, @intCast(tokens[n])) == null) n += 1;
    const keep = @min(n + 1, lane.drafts);
    for (0..keep) |k| lane.ids[1 + k] = @intCast(tokens[k]);
    if (keep < lane.drafts and lane.law.ctx != null) lane.law = try g.sliceAxis(lane.law, 0, 0, @intCast(keep));
    lane.drafts = keep;
}

/// The second half (`NativeDrafter.finish`, `finish_group`): keeps each slot's `t1` and the first `accepted` drafts of
/// the open verify, `slots` in the order `verifyGroup` took them (null: the one slot).
fn finishGroup(m: *Loaded, slots: ?[]const u32, accepted: []const u32, outs: []abi.ForwardOut) !void {
    const o = m.opened orelse return error.Inkling2NoOpenVerify;
    m.opened = null;
    defer closeOpen(o);
    if (accepted.len != o.n) return error.Inkling2NoOpenVerify;
    if (slots) |s| for (s, o.lanes[0..o.n]) |slot, lane| if (slot != lane.slot) return error.Inkling2NoOpenVerify;
    for (accepted, outs, 0..) |a, *out, i| out.round.position = try commitLane(m, o, i, a);
}

/// A lane's verify rows (the nodes of its draft tree) by the round's lanes and whether one of them samples, with the
/// tensor unit and without it. Every row reads its experts' banks, so more lanes take fewer rows each, and a sampled
/// lane, which keeps fewer of its rows, fewer again.
pub fn treeWidth(n: usize, sampled: bool) u32 {
    if (!device.naxAvailable()) return if (n == 1) 8 else if (n <= 4) 5 else 4;
    if (sampled) return if (n <= 2) 6 else if (n <= 4) 5 else 4;
    return if (n <= 2) 8 else 6;
}

/// Verify rows per lane for a grouped bundled-drafter round (`stepGroupBundled`), picking the width `2..block`
/// with the highest expected tokens/ms under the group's empirical acceptance rate `acc` and packed-bank cost curve.
pub fn bundledGroupRows(n: usize, block_len: u32, acc: f32) u32 {
    const p = std.math.clamp(acc, 0.05, 0.99);
    const max_w: u32 = @min(block_len, @max(@as(u32, 2), @as(u32, @intCast(kernels.packed_rows)) / @as(u32, @intCast(@max(n, 1)))));
    var best_w: u32 = 2;
    var best_rate: f32 = 0;
    var expected: f32 = 1.0;
    var alive: f32 = 1.0;
    var w: u32 = 2;
    while (w <= max_w) : (w += 1) {
        alive *= p;
        expected += alive;
        const total_rows: f32 = @floatFromInt(n * @as(usize, w));
        const drafts: f32 = @floatFromInt(w - 1);
        const ms = 24.5 + 0.75 * total_rows + 0.35 * drafts;
        const rate = expected / ms;
        if (rate > best_rate * 1.01) {
            best_rate = rate;
            best_w = w;
        }
    }
    return best_w;
}

/// One batched bundled-drafter round for `2..max_open` lanes (`NativeDrafter.step_group`), greedy, sampled or mixed:
/// drafts all lanes in one batched pass, runs verify + argmax or GPU rejection sampling in one sync; copy groups decline.
fn stepGroupBundled(m: *Loaded, fs: []const abi.Forward, outs: []abi.ForwardOut) !abi.Verdict {
    const d1 = m.drafter orelse return abi.Verdict.refuse(.unsupported, "no bundled drafter");
    const n = fs.len;
    if (n < 2 or !mlx.streamIsGpu(m.base.s) or !kernels.enabled())
        return abi.Verdict.refuse(.unsupported, "bundled solo or cpu");
    const w = &m.base.weights;
    const cfg = &w.body.cfg;
    const sp = w.spec orelse return abi.Verdict.refuse(.unsupported, "no spec state");
    if (m.opened) |x| {
        closeOpen(x);
        m.opened = null;
    }

    var slot_ids: [max_open]u32 = undefined;
    var positions: [max_open]u32 = undefined;
    var handles: [max_open]abi.Handle = undefined;
    var t1_ids: [max_open]i32 = undefined;
    var stops: [16]u32 = undefined;
    var n_stops: usize = 0;
    var sampled = false;
    var max_cap: u32 = 0;
    var hit_sum: f32 = 0;
    var seen_sum: f32 = 0;
    var grp_hit_sum: f32 = 0;
    var grp_seen_sum: f32 = 0;
    for (fs, 0..) |*f, i| {
        const given: []const u32 = if (f.size >= @offsetOf(abi.Forward, "draft_ids") + @sizeOf(abi.Span(u32))) f.draft_ids.slice() else &.{};
        if (given.len > 0) return abi.Verdict.refuse(.unsupported, "copy drafts in group");
        const slot = f.slots.slice()[0];
        const pos = f.positions.slice()[0];
        sp.select(slot);
        if (sp.rows(pos) == 0 or f.accept_cap == 0) return abi.Verdict.refuse(.unsupported, "lane needs plain step");
        for (sp.calib.hit, sp.calib.seen) |h, se| {
            hit_sum += h;
            seen_sum += se;
        }
        grp_hit_sum += sp.group_hit;
        grp_seen_sum += sp.group_seen;
        max_cap = @max(max_cap, f.accept_cap);
        slot_ids[i] = slot;
        positions[i] = pos;
        handles[i] = f.caches.slice()[0];
        t1_ids[i] = @intCast(f.tokens.slice()[0]);
        for (stopIds(f)) |sid| {
            if (n_stops < stops.len and std.mem.indexOfScalar(u32, stops[0..n_stops], sid) == null) {
                stops[n_stops] = sid;
                n_stops += 1;
            }
        }
        sampled = sampled or (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64) and f.draft_temperature > 0);
    }
    var g = ops.Graph.init(gpa, m.base.s);
    defer g.deinit();
    if (sampled and !kernels.markovRuns(&g, d1.rnn.w, d1.fold, @intCast(d1.cfg.rank)))
        return abi.Verdict.refuse(.unsupported, "sampled bundled needs markov kernel");

    const acc_rate: f32 = if (sampled)
        (grp_hit_sum + 6.0 * 0.65) / (grp_seen_sum + 6.0)
    else
        (hit_sum + 6.0 * 0.72) / (seen_sum + 6.0);
    const target_rows = bundledGroupRows(n, d1.cfg.block, acc_rate);
    const drafts: u32 = @min(target_rows - 1, @min(d1.cfg.block - 1, max_cap));
    if (drafts == 0) return abi.Verdict.refuse(.unsupported, "zero drafts");
    const w_rows: u32 = 1 + drafts;

    const t1_mat = try g.fromSlice(i32, t1_ids[0..n], &.{ @intCast(n), 1 });
    const erun: run_mod.Run = .{ .mode = .verify, .rows = 1, .slots = @intCast(n), .positions = positions[0..n], .caches = handles[0..n], .cache_ops = fs[0].cache_ops };
    var ec = erun.ctx(&g, &m.base.shared);
    const anchors = try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &ec, t1_mat), w.embed_norm, cfg.rms_norm_eps);

    var contexts: [max_open]draft.ContextKv = undefined;
    var draft_lanes: [max_open]draft.Drafter.DraftLane = undefined;
    var seeds: [max_open]u64 = undefined;
    var sample_lanes: [max_open]kernels.SampleLane = undefined;
    for (fs, 0..) |*f, i| {
        sp.select(slot_ids[i]);
        contexts[i] = sp.context;
        const at: c_int = @intCast(i);
        draft_lanes[i] = .{
            .t1 = try g.reshape(try g.sliceAxis(t1_mat, 0, at, at + 1), &.{1}),
            .anchor = try g.sliceAxis(anchors, 0, at, at + 1),
            .taps = sp.ring,
            .position = @intCast(positions[i]),
            .cache = &contexts[i],
            .steps = drafts,
        };
        const temp: f32 = if (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64) and f.draft_temperature > 0) f.draft_temperature else 0.0;
        const seed: u64 = if (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64)) f.draft_seed else 0;
        const top_k: u32 = if (f.size >= @offsetOf(abi.Forward, "top_k") + @sizeOf(u32)) f.top_k else 0;
        const top_p: f32 = if (f.size >= @offsetOf(abi.Forward, "top_p") + @sizeOf(f32)) f.top_p else 1.0;
        const min_p: f32 = if (f.size >= @offsetOf(abi.Forward, "min_p") + @sizeOf(f32)) f.min_p else 0.0;
        seeds[i] = seed;
        sample_lanes[i] = .{
            .temperature = temp,
            .top_k = top_k,
            .top_p = top_p,
            .min_p = min_p,
            .seed = seed ^ 0x9E3779B97F4A7C15,
            .accept_cap = @min(f.accept_cap, drafts),
        };
    }

    var packed_b: draft.Drafter.BatchBlock = .{ .tokens = .{}, .confidence = .{} };
    var sampled_b: draft.Drafter.SampledBlock = .{ .tokens = .{}, .cands = .{}, .probs = .{}, .draws = .{}, .k = 0 };
    var all_tokens: ops.Array = .{};
    var parent: ops.Array = .{};
    var tmeta: ops.Array = .{};
    var next_q: ops.Array = .{};
    var draws_full: ops.Array = .{};
    const all_ids = if (!sampled) blk: {
        packed_b = try d1.draftBatchPacked(&g, draft_lanes[0..n], &w.head.lin, w.head.vocab, w.head.softcap, null);
        try ops.asyncMany(&.{ packed_b.tokens, packed_b.confidence });
        break :blk try g.reshape(try g.concat(&.{ t1_mat, packed_b.tokens }, 1), &.{@intCast(n * w_rows)});
    } else blk: {
        sampled_b = try d1.draftBatchSampled(&g, draft_lanes[0..n], sample_lanes[0..n], seeds[0..n], &w.head.lin, w.head.vocab, w.head.softcap);
        const b_i: c_int = @intCast(n);
        const w_i: c_int = @intCast(w_rows);
        all_tokens = try g.concat(&.{ t1_mat, sampled_b.tokens }, 1);
        var par_buf: [max_open * kernels.open_rows]i32 = undefined;
        var meta_buf: [max_open * kernels.open_rows * 2]i32 = undefined;
        for (0..n) |i| for (0..w_rows) |r_idx| {
            par_buf[i * w_rows + r_idx] = @as(i32, @intCast(r_idx)) - 1;
            meta_buf[(i * w_rows + r_idx) * 2 + 0] = @intCast(r_idx);
            meta_buf[(i * w_rows + r_idx) * 2 + 1] = 0;
        };
        parent = try g.fromSlice(i32, par_buf[0 .. n * w_rows], &.{ b_i, w_i });
        tmeta = try g.fromSlice(i32, meta_buf[0 .. n * w_rows * 2], &.{ b_i * w_i, 2 });
        next_q = try g.concat(&.{ sampled_b.probs, try g.zeros(&.{ b_i, 1, sampled_b.k }, .float32) }, 1);
        const ff_leaf = try g.broadcast(try g.scalar(255, .uint8), &.{ b_i, 1 });
        const d0 = try g.expandDims(try g.concat(&.{ sampled_b.draws, ff_leaf }, 1), 2);
        const ff_pad = try g.broadcast(try g.scalar(255, .uint8), &.{ b_i, w_i, 3 });
        draws_full = try g.concat(&.{ d0, ff_pad }, 2);
        try ops.asyncMany(&.{ all_tokens, sampled_b.cands, next_q, draws_full });
        break :blk try g.reshape(all_tokens, &.{@intCast(n * w_rows)});
    };
    for (0..n) |i| {
        sp.select(slot_ids[i]);
        sp.context = contexts[i];
    }

    const opens = try gpa.alloc(kernels.Open, cfg.visits());
    defer gpa.free(opens);
    @memset(opens, .{ .g = &g });
    sp.opens = opens;
    defer sp.opens = null;

    var fo: abi.ForwardOut = .{};
    defer {
        if (fo.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.logits.ctx });
        if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
    }
    m.split.follow(false);
    const v = run_mod.tableForward(Model, &m.base, &.{
        .mode = .batched_rows,
        .logits = .all_rows,
        .rows = w_rows,
        .tokens = .{},
        .ids = .{ .ctx = all_ids.ctx },
        .slots = .of(slot_ids[0..n]),
        .positions = .of(positions[0..n]),
        .caches = .of(handles[0..n]),
        .cache_ops = fs[0].cache_ops,
        .pipeline_every = pipeline_layers,
    }, &fo);
    const held = sp.held;
    sp.held = .{};
    defer if (held.ctx != null) {
        _ = mlx.mlx_array_free(held);
    };
    if (!v.isOk()) return v;

    var rings: [max_open]ops.Array = undefined;
    if (!sampled) {
        const flat_logits = try g.reshape(.{ .ctx = fo.logits.ctx }, &.{ @intCast(n * w_rows), -1 });
        const picked = try picks(&g, flat_logits, fs[0].suppress_mask);
        try ops.evalMany(&.{ packed_b.tokens, packed_b.confidence, picked });
        const draft_data = mlx.mlx_array_data_int32(packed_b.tokens) orelse return error.Inkling2DraftRound;
        const sure_data = mlx.mlx_array_data_float32(packed_b.confidence) orelse return error.Inkling2DraftRound;
        const want_data = mlx.mlx_array_data_int32(picked) orelse return error.Inkling2DraftRound;

        for (fs, 0..) |*f, i| {
            const lane_stops = stopIds(f);
            const lane_drafts = draft_data[i * drafts .. (i + 1) * drafts];
            const lane_sure = sure_data[i * drafts .. (i + 1) * drafts];
            const lane_want = want_data[i * w_rows .. (i + 1) * w_rows];
            const lane_cap = @min(drafts, f.accept_cap);
            var accepted: u32 = 0;
            while (accepted < lane_cap) : (accepted += 1) {
                const dtok: u32 = @intCast(lane_drafts[accepted]);
                if (std.mem.indexOfScalar(u32, lane_stops, dtok) != null) break;
                if (lane_want[accepted] != lane_drafts[accepted]) break;
            }
            sp.select(slot_ids[i]);
            for (lane_sure[0..@min(lane_cap, accepted + 1)], 0..) |conf, k| sp.calib.observe(conf, k < accepted);
            const kept: c_int = @intCast(1 + accepted);
            const h_one = [_]abi.Handle{handles[i]};
            const p_one = [_]u32{positions[i]};
            const crun: run_mod.Run = .{ .mode = .verify, .rows = kept, .positions = &p_one, .caches = &h_one, .cache_ops = fs[0].cache_ops };
            var cc = crun.ctx(&g, &m.base.shared);
            try kernels.commit(&cc, cfg, opens, kept, i, n);
            const at: c_int = @intCast(i * w_rows);
            try sp.pushOpt(&g, positions[i], &.{try g.sliceAxis(held, 1, at, at + kept)}, false);
            rings[i] = sp.ring;

            outs[i].round = .{
                .n = @intCast(kept),
                .accepted = accepted,
                .next_token = @intCast(lane_want[accepted]),
                .rows = w_rows,
                .position = @as(u64, positions[i]) + @as(u64, @intCast(kept)),
            };
            outs[i].round.tokens[0] = @intCast(t1_ids[i]);
            for (0..accepted) |k| outs[i].round.tokens[1 + k] = @intCast(lane_drafts[k]);
        }
    } else {
        const result = try kernels.sampleAndAcceptTree(
            &g,
            .{ .ctx = fo.logits.ctx },
            w.head.vocab,
            .{ .ctx = fs[0].suppress_mask.ctx },
            all_tokens,
            parent,
            sample_lanes[0..n],
            @intCast(w_rows),
            stops[0..n_stops],
            .{ .tmeta = tmeta, .cands = sampled_b.cands, .next_q = next_q, .draws = draws_full, .r = @intCast(drafts), .k = sampled_b.k },
        );
        try ops.eval(result);
        const res_data = mlx.mlx_array_data_int32(result) orelse return error.Inkling2DraftRound;
        const stride = 2 + 2 * @as(usize, w_rows);
        for (fs, 0..) |*f, i| {
            const row = res_data[i * stride .. (i + 1) * stride];
            const accepted: u32 = @intCast(row[0]);
            const next_token: u32 = @intCast(row[1]);
            const kept: usize = 1 + @as(usize, accepted);
            if (kept > w_rows or kept > abi.max_round_tokens) return error.Inkling2DraftRound;

            sp.select(slot_ids[i]);
            const lane_cap = @min(drafts, f.accept_cap);
            const tested = @min(lane_cap, accepted + 1);
            sp.group_hit = sp.group_hit * 0.92 + @as(f32, @floatFromInt(accepted));
            sp.group_seen = sp.group_seen * 0.92 + @as(f32, @floatFromInt(tested));

            const h_one = [_]abi.Handle{handles[i]};
            const p_one = [_]u32{positions[i]};
            const crun: run_mod.Run = .{ .mode = .verify, .rows = @intCast(kept), .positions = &p_one, .caches = &h_one, .cache_ops = fs[0].cache_ops };
            var cc = crun.ctx(&g, &m.base.shared);
            try kernels.commit(&cc, cfg, opens, @intCast(kept), i, n);
            const at: c_int = @intCast(i * w_rows);
            try sp.pushOpt(&g, positions[i], &.{try g.sliceAxis(held, 1, at, at + @as(c_int, @intCast(kept)))}, false);
            rings[i] = sp.ring;

            outs[i].round = .{
                .n = @intCast(kept),
                .accepted = accepted,
                .next_token = next_token,
                .rows = w_rows,
                .position = @as(u64, positions[i]) + @as(u64, @intCast(kept)),
            };
            for (0..kept) |k| outs[i].round.tokens[k] = @intCast(row[2 + w_rows + k]);
        }
    }
    try ops.asyncMany(rings[0..n]);
    return .ok;
}

/// One DFlash2 round for `1..max_open` lanes, greedy, sampled or mixed (`NativeDrafter.step_group`): the draft tree, its
/// verify, the accept and the path's commit stay on the GPU, and the host reads one small result a round.
fn stepGroup(m: *Loaded, fs: []const abi.Forward, outs: []abi.ForwardOut) !abi.Verdict {
    const d2 = m.drafter2 orelse return stepGroupBundled(m, fs, outs);
    if (!env.notOff(.MLX_SERVE_INKLING2_TREE) or !mlx.streamIsGpu(m.base.s) or !kernels.enabled())
        return abi.Verdict.refuse(.unsupported, "tree verify disabled");
    const w = &m.base.weights;
    const cfg = &w.body.cfg;
    const sp = w.spec orelse return abi.Verdict.refuse(.unsupported, "no spec state");
    const n = fs.len;
    if (m.opened) |x| {
        closeOpen(x);
        m.opened = null;
    }

    var slot_ids: [max_open]u32 = undefined;
    var positions: [max_open]u32 = undefined;
    var handles: [max_open]abi.Handle = undefined;
    var t1_ids: [max_open]i32 = undefined;
    var stops: [16]u32 = undefined;
    var n_stops: usize = 0;
    var sampled = false;
    for (fs, 0..) |*f, i| {
        const slot = f.slots.slice()[0];
        const pos = f.positions.slice()[0];
        sp.select(slot);
        if (sp.rows(pos) == 0 or f.accept_cap == 0) return abi.Verdict.refuse(.unsupported, "lane needs plain step");
        slot_ids[i] = slot;
        positions[i] = pos;
        handles[i] = f.caches.slice()[0];
        t1_ids[i] = @intCast(f.tokens.slice()[0]);
        for (stopIds(f)) |sid| {
            if (n_stops < stops.len and std.mem.indexOfScalar(u32, stops[0..n_stops], sid) == null) {
                stops[n_stops] = sid;
                n_stops += 1;
            }
        }
        sampled = sampled or (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64) and f.draft_temperature > 0);
    }

    const w_rows = treeWidth(n, sampled);
    var g = ops.Graph.init(gpa, m.base.s);
    defer g.deinit();

    // Embed all lanes' anchor tokens in one batched embed + rmsNorm.
    const t1_mat = try g.fromSlice(i32, t1_ids[0..n], &.{ @intCast(n), 1 });
    const erun: run_mod.Run = .{ .mode = .verify, .rows = 1, .slots = @intCast(n), .positions = positions[0..n], .caches = handles[0..n], .cache_ops = fs[0].cache_ops };
    var ec = erun.ctx(&g, &m.base.shared);
    const anchors = try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &ec, t1_mat), w.embed_norm, cfg.rms_norm_eps);

    var contexts: [max_open]draft.ContextKv = undefined;
    var tree_lanes: [max_open]dflash2.Drafter.TreeLane = undefined;
    var sample_lanes: [max_open]kernels.SampleLane = undefined;
    for (fs, 0..) |*f, i| {
        sp.select(slot_ids[i]);
        contexts[i] = sp.context;
        const temp: f32 = if (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64) and f.draft_temperature > 0) f.draft_temperature else 0.0;
        const seed: u64 = if (f.size >= @offsetOf(abi.Forward, "draft_seed") + @sizeOf(u64)) f.draft_seed else 0;
        const top_k: u32 = if (f.size >= @offsetOf(abi.Forward, "top_k") + @sizeOf(u32)) f.top_k else 0;
        const top_p: f32 = if (f.size >= @offsetOf(abi.Forward, "top_p") + @sizeOf(f32)) f.top_p else 1.0;
        const min_p: f32 = if (f.size >= @offsetOf(abi.Forward, "min_p") + @sizeOf(f32)) f.min_p else 0.0;
        const copy: []const u32 = if (f.size >= @offsetOf(abi.Forward, "draft_ids") + @sizeOf(abi.Span(u32))) f.draft_ids.slice() else &.{};
        const at: c_int = @intCast(i);
        tree_lanes[i] = .{
            .t1 = @intCast(t1_ids[i]),
            .anchor = if (n == 1) anchors else try g.sliceAxis(anchors, 0, at, at + 1),
            .taps = sp.ring,
            .position = @intCast(positions[i]),
            .cache = &contexts[i],
            .temperature = temp,
            .seed = seed,
            .calib_slope = sp.tree_calib.slope,
            .calib_intercepts = sp.tree_calib.intercepts,
            .copy = copy,
            .copy_rate = 0.9,
        };
        sample_lanes[i] = .{
            .temperature = temp,
            .top_k = top_k,
            .top_p = top_p,
            .min_p = min_p,
            .seed = seed ^ 0x9E3779B97F4A7C15,
            .accept_cap = @min(f.accept_cap, w_rows - 1),
        };
    }

    const tree = try d2.draftTreeBatch(&g, tree_lanes[0..n], &w.head.lin, w.head.vocab, @intCast(w_rows), 4, stops[0..n_stops]);
    // The draft runs on the GPU while the host builds the verify behind it.
    try ops.asyncMany(&.{ tree.tokens, tree.parent, tree.tmeta, tree.cands, tree.next_q, tree.draws, tree.scores });
    for (0..n) |i| {
        sp.select(slot_ids[i]);
        sp.context = contexts[i];
    }

    const opens = try gpa.alloc(kernels.Open, cfg.visits());
    defer gpa.free(opens);
    const tctx: kernels.TreeCtx = .{ .parent = tree.parent, .meta = tree.tmeta };
    for (opens) |*o| o.* = .{ .g = &g, .tree = tctx };
    sp.opens = opens;
    defer sp.opens = null;

    var fo: abi.ForwardOut = .{};
    defer {
        if (fo.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.logits.ctx });
        if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
    }
    m.split.follow(false);
    const v = run_mod.tableForward(Model, &m.base, &.{
        .mode = if (n == 1) .verify else .batched_rows,
        .logits = .all_rows,
        .rows = w_rows,
        .tokens = .{},
        .ids = .{ .ctx = tree.tokens.ctx },
        .slots = .of(slot_ids[0..n]),
        .positions = .of(positions[0..n]),
        .caches = .of(handles[0..n]),
        .cache_ops = fs[0].cache_ops,
        .ablation = if (n == 1) fs[0].ablation else null,
        .pipeline_every = pipeline_layers,
    }, &fo);
    const held = sp.held;
    sp.held = .{};
    defer if (held.ctx != null) {
        _ = mlx.mlx_array_free(held);
    };
    if (!v.isOk()) return v;

    const result = try kernels.sampleAndAcceptTree(
        &g,
        .{ .ctx = fo.logits.ctx },
        w.head.vocab,
        .{ .ctx = fs[0].suppress_mask.ctx },
        tree.tokens,
        tree.parent,
        sample_lanes[0..n],
        @intCast(w_rows),
        stops[0..n_stops],
        .{ .tmeta = tree.tmeta, .cands = tree.cands, .next_q = tree.next_q, .draws = tree.draws, .r = tree.r, .k = tree.k },
    );
    // Single GPU→CPU sync per round: reads ~1.5 KB (accepted paths, next tokens, tree calibration scores).
    try ops.evalMany(&.{ result, tree.scores, tree.parent });
    const res_data = mlx.mlx_array_data_int32(result) orelse return error.Inkling2DraftRound;
    const sc_data = mlx.mlx_array_data_float32(tree.scores) orelse return error.Inkling2DraftRound;
    const par_data = mlx.mlx_array_data_int32(tree.parent) orelse return error.Inkling2DraftRound;
    const stride = 2 + 2 * @as(usize, w_rows);

    for (0..n) |i| {
        const row = res_data[i * stride .. (i + 1) * stride];
        const accepted: u32 = @intCast(row[0]);
        const next_token: u32 = @intCast(row[1]);
        const kept: usize = 1 + @as(usize, accepted);
        if (kept > w_rows or kept > abi.max_round_tokens) return error.Inkling2DraftRound;

        var path: [kernels.open_rows]u32 = undefined;
        var emitted: [kernels.open_rows]u32 = undefined;
        var linear = true;
        for (0..kept) |k| {
            path[k] = @intCast(row[2 + k]);
            emitted[k] = @intCast(row[2 + w_rows + k]);
            if (path[k] != k) linear = false;
        }

        sp.select(slot_ids[i]);
        // Update per-slot online logistic calibration along accepted path + top rejected child if any.
        const sc_lane = sc_data[i * w_rows .. (i + 1) * w_rows];
        const par_lane = par_data[i * w_rows .. (i + 1) * w_rows];
        for (1..kept) |k| {
            const q = sc_lane[path[k]];
            if (q > 0.0) sp.tree_calib.observe(k - 1, q, true);
        }
        const last_row: i32 = @intCast(path[kept - 1]);
        var best_rej_q: f32 = -1.0;
        for (1..w_rows) |r_idx| {
            if (par_lane[r_idx] == last_row and sc_lane[r_idx] > best_rej_q) best_rej_q = sc_lane[r_idx];
        }
        if (best_rej_q > 0.0) sp.tree_calib.observe(kept - 1, best_rej_q, false);

        const h_one = [_]abi.Handle{handles[i]};
        const p_one = [_]u32{positions[i]};
        const crun: run_mod.Run = .{ .mode = .verify, .rows = @intCast(kept), .positions = &p_one, .caches = &h_one, .cache_ops = fs[0].cache_ops };
        var cc = crun.ctx(&g, &m.base.shared);
        try kernels.commitPath(&cc, cfg, opens, path[0..kept], i, n);

        const at: c_int = @intCast(i * w_rows);
        const lane_taps = if (linear)
            try g.sliceAxis(held, 1, at, at + @as(c_int, @intCast(kept)))
        else blk: {
            var tidx: [kernels.open_rows]i32 = undefined;
            for (0..kept) |k| tidx[k] = at + @as(i32, @intCast(path[k]));
            break :blk try g.take(held, try g.fromSlice(i32, tidx[0..kept], &.{@intCast(kept)}), 1);
        };
        try sp.push(&g, positions[i], &.{lane_taps});

        outs[i].round = .{
            .n = @intCast(kept),
            .accepted = accepted,
            .next_token = next_token,
            .rows = w_rows,
            .position = @as(u64, positions[i]) + @as(u64, @intCast(kept)),
        };
        @memcpy(outs[i].round.tokens[0..kept], emitted[0..kept]);
    }
    return .ok;
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
    /// The DFlash2 draft beside the pack, which serves in place of the bundled one.
    drafter2: ?*dflash2.Drafter = null,
    /// A sampled round's verify, of one slot or a group, open until its finish.
    opened: ?*Opened = null,
    spec: Spec = .{ .visits = &.{}, .keep = 0 },
    tap_visits: [config.max_taps]u32 = @splat(0),
    /// The host's command-buffer split, moved per forward (`BufferSplit`).
    split: BufferSplit = .{},

    /// The block the loaded drafter proposes (the anchor and its masks); 0 = none.
    fn blockWidth(m: *const Loaded) u32 {
        if (m.drafter2) |d| return d.cfg.block;
        if (m.drafter) |d| return d.cfg.block;
        return 0;
    }

    fn release(m: *Loaded) void {
        if (m.tower) |*t| t.deinit();
        if (m.opened) |o| closeOpen(o);
        if (m.drafter) |d| d.deinit();
        if (m.drafter2) |d| d.deinit();
        m.spec.deinit();
        m.owned.release();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.split.follow(false);
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (f.slots.len == 1) {
        const slot = f.slots.slice()[0];
        const from: u32 = if (f.size >= @offsetOf(abi.Forward, "adopt_len") + @sizeOf(u64)) f.adopt_slot else 0;
        if (from != 0 and from != slot) m.spec.adopt(from, slot, f.adopt_len) else m.spec.select(slot);
    }
    m.split.follow(f.rows == 1);
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

    /// `loader.join` with zero rows behind the parts up to a multiple of 64: the tensor unit's quantized matmul kernels
    /// take an output width of whole tiles only, and a verify's rows fell to a slow lane without them. `widths` name
    /// the parts alone, so a reader never slices the pad.
    fn joinQuant(a: At, parts: []const *Linear) !b.linear.Joined {
        const q0 = parts[0].*;
        for (parts) |p| if (p.s.ctx == null or p.bits != q0.bits or p.group_size != q0.group_size or p.mode != q0.mode or (p.b.ctx == null) != (q0.b.ctx == null) or ops.dtype(p.w) != ops.dtype(q0.w) or ops.dtype(p.s) != ops.dtype(q0.s)) {
            return loader.join(a.o, parts, a.r.s);
        };
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        var triples: [3][5]Array = undefined;
        var j: b.linear.Joined = .{};
        var sum: c_int = 0;
        for (parts, 0..) |p, i| {
            triples[0][i] = p.w;
            triples[1][i] = p.s;
            triples[2][i] = p.b;
            j.widths[i] = ops.shape(p.w)[0];
            sum += j.widths[i];
        }
        const pad = @rem(64 - @rem(sum, 64), 64);
        var count = parts.len;
        if (pad > 0) {
            for (&triples, [_]Array{ q0.w, q0.s, q0.b }) |*t, like| if (like.ctx != null) {
                t[count] = try g.zeros(&.{ pad, ops.shape(like)[1] }, ops.dtype(like));
            };
            count += 1;
        }
        var fused: [3]Array = @splat(.{});
        for (&fused, &triples, [_]Array{ q0.w, q0.s, q0.b }) |*f, *t, like| {
            if (like.ctx != null) f.* = try a.keep(&g, try g.contiguous(try g.concat(t[0..count], 0)));
        }
        var start: c_int = 0;
        for (parts, j.widths[0..parts.len]) |p, n| {
            for ([_]*Array{ &p.w, &p.s, &p.b }, fused) |x, f| if (f.ctx != null) {
                _ = mlx.mlx_array_free(x.*);
                x.* = g.escape(try g.sliceAxis(f, 0, start, start + n));
            };
            start += n;
        }
        j.lin = .{ .w = fused[0], .s = fused[1], .b = fused[2], .bits = q0.bits, .group_size = q0.group_size, .mode = q0.mode };
        return j;
    }

    /// Six quantized projections `[k, v, q, gate, r, sink]` joined in one 64-row-aligned buffer so `join_qkv`,
    /// `join_kv` (`[0..kv_rows]`) and `join_q` (`[kv_rows..total]`) share storage with zero extra memory when
    /// `kv_rows` is already a multiple of 64.
    fn joinQuant6(a: At, parts: [6]*Linear) !?struct { qkv: layers.Joined6, kv: b.linear.Joined, q: b.linear.Joined } {
        const q0 = parts[0].*;
        for (parts) |p| if (p.s.ctx == null or p.bits != q0.bits or p.group_size != q0.group_size or p.mode != q0.mode or (p.b.ctx == null) != (q0.b.ctx == null) or ops.dtype(p.w) != ops.dtype(q0.w) or ops.dtype(p.s) != ops.dtype(q0.s)) {
            return null;
        };
        const kv_rows = ops.shape(parts[0].w)[0] + ops.shape(parts[1].w)[0];
        if (@rem(kv_rows, 64) != 0) return null;
        var g = ops.Graph.init(gpa, a.r.s);
        defer g.deinit();
        var triples: [3][7]Array = undefined;
        var j6: layers.Joined6 = .{};
        var sum: c_int = 0;
        for (parts, 0..) |p, i| {
            triples[0][i] = p.w;
            triples[1][i] = p.s;
            triples[2][i] = p.b;
            j6.widths[i] = ops.shape(p.w)[0];
            sum += j6.widths[i];
        }
        const pad = @rem(64 - @rem(sum, 64), 64);
        var count: usize = 6;
        if (pad > 0) {
            for (&triples, [_]Array{ q0.w, q0.s, q0.b }) |*t, like| if (like.ctx != null) {
                t[count] = try g.zeros(&.{ pad, ops.shape(like)[1] }, ops.dtype(like));
            };
            count += 1;
        }
        var fused: [3]Array = @splat(.{});
        for (&fused, &triples, [_]Array{ q0.w, q0.s, q0.b }) |*f, *t, like| {
            if (like.ctx != null) f.* = try a.keep(&g, try g.contiguous(try g.concat(t[0..count], 0)));
        }
        const total = sum + pad;
        var start: c_int = 0;
        for (parts, j6.widths) |p, n| {
            for ([_]*Array{ &p.w, &p.s, &p.b }, fused) |x, f| if (f.ctx != null) {
                _ = mlx.mlx_array_free(x.*);
                x.* = g.escape(try g.sliceAxis(f, 0, start, start + n));
            };
            start += n;
        }
        j6.lin = .{ .w = fused[0], .s = fused[1], .b = fused[2], .bits = q0.bits, .group_size = q0.group_size, .mode = q0.mode };
        var jkv: b.linear.Joined = .{ .widths = .{ j6.widths[0], j6.widths[1], 0, 0 } };
        var jq: b.linear.Joined = .{ .widths = .{ j6.widths[2], j6.widths[3], j6.widths[4], j6.widths[5] } };
        jkv.lin = .{ .bits = q0.bits, .group_size = q0.group_size, .mode = q0.mode };
        jq.lin = .{ .bits = q0.bits, .group_size = q0.group_size, .mode = q0.mode };
        for ([_]*Array{ &jkv.lin.w, &jkv.lin.s, &jkv.lin.b }, [_]*Array{ &jq.lin.w, &jq.lin.s, &jq.lin.b }, fused) |kvd, qd, f| {
            if (f.ctx != null) {
                kvd.* = try a.keep(&g, try g.sliceAxis(f, 0, 0, kv_rows));
                qd.* = try a.keep(&g, try g.sliceAxis(f, 0, kv_rows, total));
            }
        }
        return .{ .qkv = j6, .kv = jkv, .q = jq };
    }

    fn conv(a: At, comptime x: []const u8, global: bool, hidden: u32) !layers.Conv {
        const leaf = "self_attn." ++ x ++ "_sconv";
        const rel = "attn." ++ x ++ "_sconv";
        if (!global) return .{ .static = try a.kernel(leaf ++ ".conv1d.weight", rel ++ ".weight") };
        return .{
            .dynamic = .{
                .static = try a.need(leaf ++ ".static_weight", rel ++ ".static_w"),
                // The release's is a bare `[out, in]` tensor, with no `.weight` leaf for the format to find.
                .proj = if (a.release) .{ .w = try a.need("", rel ++ ".weight_proj") } else try a.proj(leaf ++ ".weight_proj", "", hidden),
            },
        };
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
    var consts_g = ops.Graph.init(gpa, r.s);
    defer consts_g.deinit();
    const kt: c_int = @intCast(c.sconv_kernel);
    const cs_static: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1)));
    const cs_dyn: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(2 * kt + 1)));
    const attn_sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    const consts: layers.Consts = .{
        .eps = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{c.rms_norm_eps}, &.{1})),
        .caps = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{ 1.0 / c.gate_softcap, c.gate_softcap, 1.0 / c.up_softcap, c.up_softcap }, &.{4})),
        .route_scale = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{c.route_scale}, &.{1})),
        .conv_scale = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{cs_static}, &.{1})),
        .mid_pr = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{ cs_static, c.rms_norm_eps }, &.{2})),
        .prep_pr_static = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{ c.qk_clip, c.rel_clip, c.rms_norm_eps, 1.0, cs_static, 0.0 }, &.{6})),
        .prep_pr_dyn = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{ c.qk_clip, c.rel_clip, c.rms_norm_eps, 1.0, cs_dyn, 0.0 }, &.{6})),
        .attn_scale = try at.keep(&consts_g, try consts_g.scalar(attn_sc, .float32)),
        .rel_params = try at.keep(&consts_g, try consts_g.fromSlice(f32, &.{ attn_sc, c.log_alpha }, &.{2})),
        .zero_i32 = try at.keep(&consts_g, try consts_g.zeros(&.{1}, .int32)),
        .zero_tmeta = try at.keep(&consts_g, try consts_g.zeros(&.{ 1, 2 }, .int32)),
    };
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
            .consts = consts,
        },
    };
    if (c.vision) |v| m.tower = try vision.Tower.load(ctx, v, c.block_json);
    const drafts_on = draftsOn(device.deviceGeneration().gen, config.fusedOn());
    if (drafts_on and c.dflash2 != null) {
        const dc = c.dflash2.?;
        var dir: [std.fs.max_path_bytes]u8 = undefined;
        const d = try dflash2.Drafter.load(try std.fmt.bufPrint(&dir, "{s}/{s}", .{ ctx.model_dir.slice(), config.dflash2_dir }), dc, c.hidden_size, &head, r.s, draft_bits);
        m.drafter2 = d;
        // The mask token's row as the anchor's: through the table, then its norm.
        var g = ops.Graph.init(gpa, r.s);
        defer g.deinit();
        const pos = [_]u32{0};
        const run: run_mod.Run = .{ .mode = .decode, .rows = 1, .positions = &pos };
        var cx = run.ctx(&g, &m.base.shared);
        const ids = try g.fromSlice(i32, &.{@intCast(dc.mask_id)}, &.{ 1, 1 });
        try d.setMask(&g, try g.rmsNorm(try b.embed.TableEmbed.forward(&m.base.weights.embed, &cx, ids), m.base.weights.embed_norm, c.rms_norm_eps));
        m.tap_visits = dc.taps;
        m.spec = .{ .visits = m.tap_visits[0 .. dc.tap_count - 1], .keep = @intCast(dc.window - 1) };
        m.base.weights.spec = &m.spec;
    } else if (drafts_on) if (c.draft) |dc| if (try draft.Drafter.load(ctx, dc, c.hidden_size, &head, r.s)) |d| {
        m.drafter = d;
        m.tap_visits = dc.taps;
        m.spec = .{ .visits = m.tap_visits[0 .. dc.tap_count - 1], .keep = @intCast(dc.window - 1) };
        if (dc.fullBytesPerToken() > 0) m.spec.full = d;
        m.base.weights.spec = &m.spec;
    };
    // The parse's quant block is freed with it; the forward reads none of it.
    m.base.weights.body.cfg.block_json = "{}";
    m.split = BufferSplit.of(ctx.host);

    return m;
}

/// The width the DFlash2 draft's projections are held at: its weights are read every round, and drafts are only proposals.
const draft_bits = 8;

/// MLX's command-buffer MB split for a one-row forward: an expert launch binds its whole bank, so at the device's split
/// every bank launch commits and the step's encode stalls on MLX's cap of in-flight buffers.
pub const command_buffer_mb: i32 = 200;

/// The host's MB split as the load found it, which the plugin moves per forward (`follow`) and puts back at unload;
/// nothing to do on a host without the member.
const BufferSplit = struct {
    gpu: ?*const sdk_host.abi.HostGpu = null,
    prev_mb: i32 = 0,
    lifted: bool = false,

    fn of(raw: ?*const anyopaque) BufferSplit {
        const h = sdk_host.Host.of(raw) catch return .{};
        const set = h.commandBufferLimits() orelse return .{};
        var prev_ops: i32 = 0;
        var prev_mb: i32 = 0;
        // Below 1 keeps a value: this reads the pair.
        if (!set(h.gpu().ctx, 0, 0, &prev_ops, &prev_mb).isOk()) return .{};
        return .{ .gpu = h.gpu(), .prev_mb = prev_mb };
    }

    /// The lifted split for a one-row forward, the host's for a wider one.
    fn follow(sp: *BufferSplit, step: bool) void {
        const g = sp.gpu orelse return;
        if (step == sp.lifted) return;
        var ops_now: i32 = 0;
        var mb_now: i32 = 0;
        if (g.command_buffer_limits(g.ctx, 0, if (step) command_buffer_mb else sp.prev_mb, &ops_now, &mb_now).isOk()) sp.lifted = step;
    }
};

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
    var join_q: b.linear.Joined = .{};
    var join_kv: b.linear.Joined = .{};
    var join_qkv: layers.Joined6 = .{};
    if (!global and quantized) {
        if (try at.joinQuant6(.{ &lins[4], &lins[5], &lins[0], &lins[1], &lins[2], &lins[3] })) |j6| {
            join_qkv = j6.qkv;
            join_kv = j6.kv;
            join_q = j6.q;
        }
    }
    if (!join_q.built()) {
        join_q = if (quantized) try at.joinQuant(&.{ &lins[0], &lins[1], &lins[2], &lins[3] }) else try at.joinRows(&.{ &lins[0], &lins[1], &lins[2], &lins[3] });
        join_kv = if (global) b.linear.Joined{} else if (quantized) try at.joinQuant(&.{ &lins[4], &lins[5] }) else try at.joinRows(&.{ &lins[4], &lins[5] });
    }
    for (lins) |l| try at.o.linear(l);

    const rel_proj = try at.need("self_attn.rel_logits_proj.proj", "attn.rel_logits_proj.proj");
    var g = ops.Graph.init(gpa, at.r.s);
    defer g.deinit();
    const extent = ops.shape(rel_proj)[1];
    const rel_proj_t = try at.keep(&g, try g.contiguous(try g.transpose(rel_proj, &.{ 1, 0 })));
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
        .rel_proj_t = rel_proj_t,
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
        .join_qkv = join_qkv,
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
    _ = dflash2;
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
    try sp.push(&g, 10, &.{row});
    try sp.push(&g, 11, &.{row});
    sp.select(9);
    try std.testing.expectEqual(@as(c_int, 0), sp.rows(12));
    try sp.push(&g, 12, &.{row});
    sp.select(7);
    try std.testing.expectEqual(@as(c_int, 2), sp.rows(12));
    sp.select(9);
    try std.testing.expectEqual(@as(c_int, 1), sp.rows(13));
    // The next slots park the ones before them in the free places; the last takes the place of 7, parked longest.
    for (1..max_open) |id| {
        sp.select(@intCast(100 + id));
        try sp.push(&g, 0, &.{row});
    }
    sp.select(9);
    try std.testing.expectEqual(@as(c_int, 1), sp.rows(13));
    sp.select(7);
    try std.testing.expectEqual(@as(c_int, 0), sp.rows(12));
}

// Bar: the drafter loads from an M3 on (and where the generation is unknown), never on an M1 (g13) or M2 (g14).
// Bar: a new slot that adopts an earlier one's ring continues it from a position their sequences share, the rows after
// it dropped; a push past what they share, or a ring the adopter never had, restarts.
test "inkling2 draft taps: an adopted ring continues from the positions its sequence shares" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var sp: Spec = .{ .visits = &.{}, .keep = 4 };
    defer sp.deinit();
    const rows = try g.ones(&.{ 1, 3, 2 }, .float32);
    sp.select(7);
    try sp.push(&g, 0, &.{rows});
    try sp.push(&g, 3, &.{rows});
    try std.testing.expectEqual(@as(c_int, 4), sp.rows(6));
    // Slot 8 repeats 7's first five positions: the ring cut back to them, two rows past them added.
    sp.adopt(7, 8, 5);
    try sp.push(&g, 5, &.{try g.sliceAxis(rows, 1, 0, 2)});
    try std.testing.expectEqual(@as(u32, 8), sp.owner);
    try std.testing.expectEqual(@as(c_int, 4), sp.rows(7));
    // Past what the sequences share, the ring restarts.
    sp.adopt(8, 9, 3);
    try sp.push(&g, 5, &.{try g.sliceAxis(rows, 1, 0, 1)});
    try std.testing.expectEqual(@as(c_int, 1), sp.rows(6));
    // The adopted slot's ring moved: slot 8 holds none now.
    sp.select(8);
    try std.testing.expectEqual(@as(c_int, 0), sp.rows(7));
}

test "inkling2: the bundled drafter loads on Apple9 and later only" {
    for ([_]u32{ 13, 14 }) |g| try std.testing.expect(!draftsOn(g, true));
    for ([_]u32{ 0, 15, 16, 17 }) |g| try std.testing.expect(draftsOn(g, true));
    try std.testing.expect(!draftsOn(17, false));
}

// Bar: an argmax block stops before a stop id (the trunk's token carries it) and a drawn one ends on it; an untimed
// round verifies every other draft, a timed one the drafts its drafter's confidence makes worth their rows.
test "inkling2 draft: only a drawn block verifies a stop, and a timed round only the drafts worth their rows" {
    const sure = [_]f32{ 0.99, 0.99, 0.99, 0.99 };
    const tokens = [_]i32{ 5, 9, 2, 8 };
    const cal: rounds.Calibration = .{ .head = true };
    try std.testing.expectEqual(@as(u32, 4), draftsToVerify(&sure, &tokens, &.{}, &cal, &.{}));
    try std.testing.expectEqual(@as(u32, 1), draftsToVerify(&sure, &tokens, &.{ 70, 9 }, &cal, &.{}));
    try std.testing.expectEqual(@as(u32, 0), draftsToVerify(&sure, &tokens, &.{5}, &cal, &.{}));
    try std.testing.expectEqual(@as(u32, 1), draftsToVerify(&sure, &tokens, &.{5}, null, &.{}));
    try std.testing.expectEqual(@as(u32, 2), draftsToVerify(&sure, &tokens, &.{ 70, 9 }, null, &.{}));
    // A round of 13 ms with none verified, 4.5 ms over it for the first row and 1.3 ms a row after.
    const ms = [_]f32{ 13, 17.5, 18.8, 20.1, 21.4 };
    try std.testing.expectEqual(@as(u32, 4), draftsToVerify(&sure, &tokens, &.{}, &cal, &ms));
    // A doubt mid-block cuts there; a block opening on one is not verified at all.
    try std.testing.expectEqual(@as(u32, 2), draftsToVerify(&.{ 0.99, 0.99, 0.05, 0.99 }, &tokens, &.{}, &cal, &ms));
    try std.testing.expectEqual(@as(u32, 0), draftsToVerify(&.{ 0.05, 0.99, 0.99, 0.99 }, &tokens, &.{}, &cal, &ms));
    // A drawn block is verified whole.
    try std.testing.expectEqual(@as(u32, 4), draftsToVerify(&.{ 0.2, 0.2, 0.2, 0.2 }, &tokens, &.{}, null, &ms));
}

// Bar: `bundledGroupRows` stays within `2..block`, keeps `n * rows <= packed_rows`, widens with acceptance, and keeps
// 2-lane rounds wide enough (`>= 5` at 0.65 acc) not to cap speculative yield.
test "inkling2 draft: bundledGroupRows stays within packed_rows and adapts to acceptance" {
    for ([_]usize{ 2, 4, 6, 8 }) |n| {
        const low = bundledGroupRows(n, 8, 0.35);
        const mid = bundledGroupRows(n, 8, 0.70);
        const high = bundledGroupRows(n, 8, 0.90);
        try std.testing.expect(low >= 2 and low <= mid and mid <= high and high <= 8);
        try std.testing.expect(n * @as(usize, high) <= @as(usize, @intCast(kernels.packed_rows)));
    }
    try std.testing.expect(bundledGroupRows(2, 8, 0.65) >= 5);
}
