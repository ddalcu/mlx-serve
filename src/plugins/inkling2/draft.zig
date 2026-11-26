//! Inkling 2's bundled block drafter (`mtp.safetensors`, DSpark), as oMLX's `omlx/patches/inkling2/dspark.py` runs
//! it in its DFlash drafter (Apache-2.0; no other open reference ships it): a few pre-norm layers over a block of one
//! anchor (the committed token's normed embedding) and `block - 1` copies of a trained mask vector. Each layer mixes
//! the trunk's taps (the stream entering a few visits and the final-normed output) with its own softmax weights,
//! norms the mix and projects it to context keys and values; the block attends that context (the newest `window - 1`
//! committed positions) and itself in both directions, under a rotary over absolute positions. The mask rows go
//! through the trunk's LM head; an RNN Markov head adds each row a bias from the token drafted before it, so the
//! block chains left to right off one forward. Drafts only: the trunk verifies every one.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const reader = @import("../../sdk/bases/reader.zig");
const head_block = @import("../../sdk/bases/blocks/head.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const config = @import("config.zig");
const kernels = @import("kernels.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const gpa = std.heap.c_allocator;
const project = head_block.project;

pub const Layer = struct {
    in_norm: Array,
    post_norm: Array,
    /// The tap mix's norm (`dflare_norm`), f32.
    mix_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    qkv: Linear = .{},
    o: Linear,
    k_ctx: Linear,
    v_ctx: Linear,
    kv_ctx: Linear = .{},
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    gate: Linear,
    up: Linear,
    gate_up: Linear = .{},
    down: Linear,
};

fn joinRaw(at: At, names: []const []const u8, parts: []const *Linear) !Linear {
    var ws: [3]Array = undefined;
    var widths: [3]c_int = undefined;
    for (names, 0..) |leaf, i| {
        var buf: [160]u8 = undefined;
        ws[i] = try at.raw(try std.fmt.bufPrint(&buf, "{s}.weight", .{leaf}));
        widths[i] = ops.shape(ws[i])[0];
    }
    const fused = try at.keep(try at.g.contiguous(try at.g.concat(ws[0..names.len], 0)));
    var start: c_int = 0;
    for (parts, widths[0..names.len]) |p, n| {
        p.* = .{ .w = try at.keep(try at.g.sliceAxis(fused, 0, start, start + n)) };
        start += n;
    }
    return .{ .w = fused };
}

/// The drafter's layers a context cache holds.
pub const max_layers = config.max_draft_layers;
/// The width and group of the drafter's copy of a dense trunk head.
const draft_head_bits = 8;
const draft_head_group = 64;

/// The drafter's context keys and values a layer (`[1, kv_heads, rows, hd]`, normed and rotated at their positions)
/// for the positions before `end`, kept across one slot's rounds: a round projects only the rows committed since. It
/// is valid while the taps ring it was projected from continues, and is released when the ring restarts.
pub const ContextKv = struct {
    k: [max_layers]Array = @splat(.{}),
    v: [max_layers]Array = @splat(.{}),
    end: u64 = 0,
    /// A full layer's context (`Drafter.appendFull`): the positions [`full_from`, `full_end`) at the front of buffers
    /// that grow a chunk at a time.
    fk: [max_layers]Array = @splat(.{}),
    fv: [max_layers]Array = @splat(.{}),
    full_from: u64 = 0,
    full_end: u64 = 0,
    /// The buffers' growth, in positions.
    chunk: c_int = 4096,

    /// Keeps the full context of the positions before `at` and drops the ring's keys and values (re-projected from the
    /// ring at the next draft).
    pub fn cut(c: *ContextKv, at: u64) void {
        for (c.k ++ c.v) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        c.k = @splat(.{});
        c.v = @splat(.{});
        c.end = 0;
        if (at < c.full_from) {
            for (c.fk ++ c.fv) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(a);
            };
            c.fk = @splat(.{});
            c.fv = @splat(.{});
            c.full_from = 0;
            c.full_end = 0;
        } else c.full_end = @min(c.full_end, at);
    }

    pub fn release(c: *ContextKv) void {
        for (c.k ++ c.v ++ c.fk ++ c.fv) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        c.* = .{};
    }
};

pub const Drafter = struct {
    cfg: config.Draft,
    hidden: u32,
    s: mlx.mlx_stream,
    owned: std.ArrayList(Array) = .empty,
    layers: []Layer = &.{},
    norm: Array = .{},
    /// `[1, 1, hidden]`.
    mask: Array = .{},
    /// Each layer's softmax weights over the taps, then the Markov head's: `[layers + 1, taps, 1]` f32.
    alpha: Array = .{},
    /// The Markov head: the previous token's row `[vocab, rank]`, the RNN's projection of [state | that row | the
    /// block row | the newest context mix] to its gate, candidate and latent, and the latent's bias over the vocab
    /// (the trunk's head over `hidden_out`, folded once: `fold` `[rank, vocab]`, `fold_bias` `[1, 1, vocab]`).
    prev_rows: Array = .{},
    rnn: Linear = .{},
    rnn_bias: Array = .{},
    rnn_norm: Array = .{},
    fold: Array = .{},
    fold_bias: Array = .{},
    /// An 8-bit copy of a dense trunk head, which the block rows are read through (`draftHead`); null = the trunk's.
    head: ?Linear = null,

    /// The drafter beside the pack, its Markov head folded over the trunk's `head`; null when the pack ships none.
    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Draft, hidden: u32, head: *const Linear, s: mlx.mlx_stream) !?*Drafter {
        if (!ctx.dir.exists(ctx.dir.ctx, abi.Str.of(config.draft_file))) return null;
        var w = try reader.headWeights(ctx, &.{config.draft_file}, "fuse_weight", "\x00");
        defer w.deinit();
        if (w.get("fuse_weight") == null) return null;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .cfg = cfg, .hidden = hidden, .s = s };
        errdefer d.release();
        var g = Graph.init(gpa, s);
        defer g.deinit();
        var at: At = .{ .owned = &d.owned, .w = &w, .g = &g };

        if (cfg.layers > max_layers) return error.Inkling2DraftGeometry;
        d.layers = try gpa.alloc(Layer, cfg.layers);
        for (d.layers, 0..) |*L, i| {
            at.layer = @intCast(i);
            L.* = .{
                .in_norm = try at.need("input_layernorm.weight"),
                .post_norm = try at.need("post_attention_layernorm.weight"),
                .mix_norm = try at.float("dflare_norm.weight"),
                .q = .{},
                .k = .{},
                .v = .{},
                .o = try at.linear("self_attn.o_proj"),
                .k_ctx = .{},
                .v_ctx = .{},
                .q_norm = try at.need("self_attn.q_norm.weight"),
                .k_norm = try at.need("self_attn.k_norm.weight"),
                .k_norm_ctx = try at.need("self_attn.k_norm_ctx.weight"),
                .gate = .{},
                .up = .{},
                .down = try at.linear("mlp.down_proj"),
            };
            L.qkv = try joinRaw(at, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, &.{ &L.q, &L.k, &L.v });
            L.kv_ctx = try joinRaw(at, &.{ "self_attn.k_proj_ctx", "self_attn.v_proj_ctx" }, &.{ &L.k_ctx, &L.v_ctx });
            L.gate_up = try joinRaw(at, &.{ "mlp.gate_proj", "mlp.up_proj" }, &.{ &L.gate, &L.up });
        }
        at.layer = null;
        d.norm = try at.need("norm.weight");
        d.mask = try at.keep(try g.reshape(try at.raw("mask_embed"), &.{ 1, 1, @intCast(hidden) }));
        const fuse = try at.raw("fuse_weight");
        const fsh = ops.shape(fuse);
        if (fsh.len != 2 or fsh[0] != @as(c_int, @intCast(cfg.layers + 1)) or fsh[1] != @as(c_int, @intCast(cfg.tap_count))) return error.Inkling2DraftGeometry;
        // As the reference: the weights rounded to the stream's dtype before they scale the f32 taps.
        d.alpha = try at.keep(try g.reshape(try g.astype(try g.astype(try g.softmax(try g.astype(fuse, .float32), -1, true), ops.dtype(d.norm)), .float32), &.{ fsh[0], fsh[1], 1 }));
        d.prev_rows = try at.need("markov_head.markov_w1.weight");
        d.rnn = try at.linear("markov_head.rnn_proj");
        d.rnn_bias = try at.need("markov_head.rnn_proj.bias");
        d.rnn_norm = try at.float("markov_head.dflare_norm.weight");
        const rank: c_int = @intCast(cfg.rank);
        if (ops.shape(d.prev_rows)[1] != rank or ops.shape(d.rnn.w)[0] != 3 * rank) return error.Inkling2DraftGeometry;
        // `head(hidden_out(latent))` = latent over `head(hidden_out's columns)` plus `head(hidden_out's bias)`.
        const out_w = try at.raw("markov_head.hidden_out.weight");
        const out_b = try at.raw("markov_head.hidden_out.bias");
        d.fold = try at.keep(try g.reshape(try project(&g, head, try g.reshape(try g.transpose(out_w, &.{ 1, 0 }), &.{ 1, rank, @intCast(hidden) })), &.{ rank, -1 }));
        d.fold_bias = try at.keep(try project(&g, head, try g.reshape(out_b, &.{ 1, 1, @intCast(hidden) })));
        d.head = try at.draftHead(head);
        return d;
    }

    fn release(d: *Drafter) void {
        for (d.owned.items) |a| _ = mlx.mlx_array_free(a);
        d.owned.deinit(gpa);
        gpa.free(d.layers);
    }

    pub fn deinit(d: *Drafter) void {
        d.release();
        gpa.destroy(d);
    }

    /// The taps `[1, n, taps * hidden]` mixed with weight row `row` and normed: `[1, n, hidden]` in the taps' dtype.
    fn mix(d: *const Drafter, g: *Graph, taps32: Array, row: c_int, weight: Array) !Array {
        const mixed = try g.sum(try g.mul(taps32, try g.sliceAxis(d.alpha, 0, row, row + 1)), 2, false);
        return g.astype(try g.rmsNorm(mixed, weight, d.cfg.eps), ops.dtype(d.norm));
    }

    /// `x` `[1, n, heads * hd]` as heads `[1, heads, n, hd]`, normed per head (null = not) and rotated from position
    /// `at` (null = not).
    fn heads(d: *const Drafter, g: *Graph, x: Array, count: u32, norm_weight: ?Array, at: ?c_int) !Array {
        const n = ops.shape(x)[1];
        var h = try g.reshape(x, &.{ 1, n, @intCast(count), @intCast(d.cfg.head_dim) });
        if (norm_weight) |w| h = try g.rmsNorm(h, w, d.cfg.eps);
        h = try g.transpose(h, &.{ 0, 2, 1, 3 });
        return if (at) |p| g.rope(h, @intCast(d.cfg.head_dim), d.cfg.theta, p) else h;
    }

    /// Appends the full layers' context keys and values of the taps at positions `first`.. to `cc` (restarted when it
    /// ends elsewhere), queued on the GPU. `parts` side by side are the taps `[1, n, taps * hidden]`: a prompt's are
    /// joined a slice of rows at a time, never whole.
    pub fn appendFull(d: *const Drafter, g: *Graph, cc: *ContextKv, parts: []const Array, first: u64) !void {
        const n = ops.shape(parts[0])[1];
        if (n == 0) return;
        const restart = cc.full_end != first;
        if (restart) cc.full_from = first;
        // Rows land in buffers grown a chunk at a time; the write is in place (the old handle is freed before it runs).
        const chunk = cc.chunk;
        const at: c_int = @intCast(first - cc.full_from);
        const need = at + n;
        var held: [2 * max_layers]Array = undefined;
        var count: usize = 0;
        const kv: c_int = @intCast(d.cfg.kv_heads);
        const hd: c_int = @intCast(d.cfg.head_dim);
        for (d.layers, 0..) |*L, i| {
            if (!d.cfg.full(i)) continue;
            var lg = g.scope();
            defer lg.deinit();
            var bufs = [2]Array{ cc.fk[i], cc.fv[i] };
            for (&bufs) |*buf| {
                if (restart and buf.ctx != null) buf.* = .{};
                const cap: c_int = if (buf.ctx != null) ops.shape(buf.*)[2] else 0;
                // Room past the rows for a draft's block keys (`draft` writes them there, then reads one view).
                const room = need + @as(c_int, @intCast(d.cfg.block));
                if (room > cap) {
                    const grown = try lg.zeros(&.{ 1, kv, @divTrunc(room + chunk - 1, chunk) * chunk, hd }, ops.dtype(d.norm));
                    buf.* = if (at > 0) try lg.sliceUpdate(grown, try lg.sliceAxis(buf.*, 2, 0, at), &.{ 0, 0, 0, 0 }, &.{ 1, kv, at, hd }) else grown;
                }
            }
            // A prompt's rows a slice at a time, so the f32 tap mix stays small.
            var r0: c_int = 0;
            while (r0 < n) : (r0 += 1024) {
                const r1 = @min(n, r0 + 1024);
                const rows32 = try lg.astype(try lg.reshape(try sideBySide(&lg, parts, r0, r1), &.{ 1, r1 - r0, @intCast(d.cfg.tap_count), @intCast(d.hidden) }), .float32);
                const context = try d.mix(&lg, rows32, @intCast(i), L.mix_norm);
                var k: Array = undefined;
                var v: Array = undefined;
                if (L.kv_ctx.w.ctx != null) {
                    const kvp = try project(&lg, &L.kv_ctx, context);
                    const kv_dim = kv * hd;
                    k = try d.heads(&lg, try lg.sliceAxis(kvp, 2, 0, kv_dim), d.cfg.kv_heads, L.k_norm_ctx, @as(c_int, @intCast(first)) + r0);
                    v = try d.heads(&lg, try lg.sliceAxis(kvp, 2, kv_dim, 2 * kv_dim), d.cfg.kv_heads, null, null);
                } else {
                    k = try d.heads(&lg, try project(&lg, &L.k_ctx, context), d.cfg.kv_heads, L.k_norm_ctx, @as(c_int, @intCast(first)) + r0);
                    v = try d.heads(&lg, try project(&lg, &L.v_ctx, context), d.cfg.kv_heads, null, null);
                }
                for ([2]Array{ k, v }, &bufs) |x, *buf| buf.* = try lg.sliceUpdate(buf.*, try lg.astype(x, ops.dtype(d.norm)), &.{ 0, 0, at + r0, 0 }, &.{ 1, kv, at + r1, hd });
            }
            for (bufs, [2]*Array{ &cc.fk[i], &cc.fv[i] }) |buf, dst| {
                const next = lg.escape(buf);
                if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
                dst.* = next;
                held[count] = next;
                count += 1;
            }
        }
        cc.full_end = first + @as(u64, @intCast(n));
        if (count == 0) return;
        const vec = mlx.mlx_vector_array_new_data(&held, count);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }

    /// Rows `r0` to `r1` of `parts`, side by side.
    pub fn sideBySide(g: *Graph, parts: []const Array, r0: c_int, r1: c_int) !Array {
        var rows: [config.max_taps]Array = undefined;
        for (parts, rows[0..parts.len]) |x, *r| r.* = try g.sliceAxis(x, 1, r0, r1);
        return if (parts.len == 1) rows[0] else g.concat(rows[0..parts.len], 2);
    }

    /// A drafted block: its tokens `[block - 1]` int32 and the drafter's probability of each `[block - 1]` f32, lazy;
    /// drawn under a `Law`, the law each was drawn from `[block - 1, vocab]` f32 and, as the confidence, its peak.
    pub const Block = struct { tokens: Array, confidence: Array, probs: Array = .{} };

    /// `steps` drafts drawn from the drafter's soft-capped softmax at `temperature`, each keyed by `seed` and its index.
    pub const Law = struct { temperature: f32, seed: u64, steps: u32 = std.math.maxInt(u32) };

    /// One lane of a batched draft call (`draftBatch`, `draftBatchPacked`).
    pub const DraftLane = struct {
        t1: Array,
        anchor: Array,
        taps: Array,
        position: c_int,
        cache: ?*ContextKv,
        steps: u32,
    };

    /// Sampled batch draw parameters: per-lane temperature and a shared base seed.
    pub const Draws = struct { temperatures: []const f32, seed: u64 };

    /// Packed output of `draftBatchPacked`: `[B, drafts]` tokens & confidence, plus optional `[B, drafts, vocab]` probs.
    pub const BatchBlock = struct { tokens: Array, confidence: Array, probs: Array = .{} };
    pub const SampledBlock = struct { tokens: Array, cands: Array, probs: Array, draws: Array, k: c_int };

    const BatchBase = struct { rows: Array, newest: Array, base: Array, t1s: Array, drafts: c_int };

    const max_lanes = 16;

    const Rope = union(enum) { none, at: c_int, each: Array };

    fn headsRope(d: *const Drafter, g: *Graph, x: Array, count: u32, norm_weight: ?Array, rope: Rope) !Array {
        const lanes = ops.shape(x)[0];
        const n = ops.shape(x)[1];
        const hd: c_int = @intCast(d.cfg.head_dim);
        var h = try g.reshape(x, &.{ lanes, n, @intCast(count), hd });
        if (norm_weight) |w| h = try g.rmsNorm(h, w, d.cfg.eps);
        h = try g.transpose(h, &.{ 0, 2, 1, 3 });
        return switch (rope) {
            .none => h,
            .at => |p| g.rope(h, hd, d.cfg.theta, p),
            .each => |p| g.ropeEach(h, hd, d.cfg.theta, p),
        };
    }

    /// Shared 3-layer drafter forward + `lm_head` over `lanes.len` (1..16) slots.
    fn draftBatchBase(d: *const Drafter, g: *Graph, lanes: []const DraftLane, head: *const Linear) !BatchBase {
        const cfg = &d.cfg;
        const count = lanes.len;
        if (count == 0 or count > max_lanes) return error.Inkling2DraftRound;
        const hidden: c_int = @intCast(d.hidden);
        const block: c_int = @intCast(cfg.block);
        const q_dim: c_int = @intCast(cfg.heads * cfg.head_dim);
        const kv_dim: c_int = @intCast(cfg.kv_heads * cfg.head_dim);
        const inter_dim: c_int = @divExact(ops.shape(d.layers[0].gate_up.w)[0], 2);
        const dt = ops.dtype(d.norm);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const back = [_]c_int{ 0, 2, 1, 3 };

        var max_steps: u32 = 1;
        for (lanes) |l| max_steps = @max(max_steps, @min(l.steps, @as(u32, @intCast(block - 1))));
        const drafts: c_int = @intCast(max_steps);

        const mask_b = try g.broadcast(try g.astype(d.mask, dt), &.{ 1, block - 1, hidden });
        var h_parts: [max_lanes]Array = undefined;
        var fresh: [max_lanes]c_int = undefined;
        var n_taps: [max_lanes]c_int = undefined;
        var rope_at: [max_lanes]i32 = undefined;
        var new32: [max_lanes]?Array = @splat(null);
        var last_taps: [max_lanes]Array = undefined;
        var t1_parts: [max_lanes]Array = undefined;
        const tapsOf = struct {
            fn f(gr: *Graph, x: Array, from: c_int, to: c_int, tap_count: u32, width: c_int) !Array {
                return gr.astype(try gr.reshape(try gr.sliceAxis(x, 1, from, to), &.{ 1, to - from, @intCast(tap_count), width }), .float32);
            }
        }.f;
        for (lanes, 0..) |lane, j| {
            const n = ops.shape(lane.taps)[1];
            n_taps[j] = n;
            rope_at[j] = lane.position;
            const at: u64 = @intCast(lane.position);
            fresh[j] = n;
            if (lane.cache) |cc| if (cc.k[0].ctx != null and cc.end <= at and at - cc.end <= @as(u64, @intCast(n))) {
                const since: c_int = @intCast(at - cc.end);
                if (since + ops.shape(cc.k[0])[2] >= n) fresh[j] = since;
            };
            if (fresh[j] > 0) new32[j] = try tapsOf(g, lane.taps, n - fresh[j], n, cfg.tap_count, hidden);
            last_taps[j] = try tapsOf(g, lane.taps, n - 1, n, cfg.tap_count, hidden);
            h_parts[j] = try g.concat(&.{ try g.astype(lane.anchor, dt), mask_b }, 1);
            t1_parts[j] = lane.t1;
        }
        var h = if (count == 1) h_parts[0] else try g.concat(h_parts[0..count], 0);
        const rope: Rope = if (count == 1) .{ .at = rope_at[0] } else .{ .each = try g.fromSlice(i32, rope_at[0..count], &.{@intCast(count)}) };

        for (d.layers, 0..) |*L, i| {
            var lg = g.scope();
            defer lg.deinit();
            var kc: [max_lanes]Array = @splat(.{});
            var vc: [max_lanes]Array = @splat(.{});
            var full_read: [max_lanes]bool = @splat(false);
            var own_rows: [max_lanes]c_int = @splat(0);
            var ctx_parts: [max_lanes]Array = undefined;
            var ctx_lanes: [max_lanes]usize = undefined;
            var need_kv: usize = 0;
            for (lanes, 0..) |lane, j| {
                const n = n_taps[j];
                const at: u64 = @intCast(lane.position);
                if (lane.cache) |cc| if (cfg.full(i) and cc.fk[i].ctx != null and cc.full_end == at) {
                    const n_full: c_int = @intCast(cc.full_end - cc.full_from);
                    kc[j] = try lg.sliceAxis(cc.fk[i], 2, 0, n_full);
                    vc[j] = try lg.sliceAxis(cc.fv[i], 2, 0, n_full);
                    full_read[j] = true;
                };
                if (!full_read[j]) {
                    const own: c_int = if (fresh[j] < n and lane.cache.?.k[i].ctx == null) n else fresh[j];
                    own_rows[j] = own;
                    if (own > 0) {
                        const rows32 = if (own == fresh[j]) new32[j].? else try tapsOf(&lg, lane.taps, 0, n, cfg.tap_count, hidden);
                        ctx_parts[need_kv] = try d.mix(&lg, rows32, @intCast(i), L.mix_norm);
                        ctx_lanes[need_kv] = j;
                        need_kv += 1;
                    }
                }
            }
            if (need_kv > 0) {
                const all_ctx = if (need_kv == 1) ctx_parts[0] else try lg.concat(ctx_parts[0..need_kv], 1);
                var k_all: Array = undefined;
                var v_all: Array = undefined;
                if (L.kv_ctx.w.ctx != null) {
                    const kvp = try project(&lg, &L.kv_ctx, all_ctx);
                    k_all = try lg.sliceAxis(kvp, 2, 0, kv_dim);
                    v_all = try lg.sliceAxis(kvp, 2, kv_dim, 2 * kv_dim);
                } else {
                    k_all = try project(&lg, &L.k_ctx, all_ctx);
                    v_all = try project(&lg, &L.v_ctx, all_ctx);
                }
                var off: c_int = 0;
                for (ctx_lanes[0..need_kv]) |j| {
                    const own = own_rows[j];
                    const kj = if (need_kv == 1) k_all else try lg.sliceAxis(k_all, 1, off, off + own);
                    const vj = if (need_kv == 1) v_all else try lg.sliceAxis(v_all, 1, off, off + own);
                    kc[j] = try d.heads(&lg, kj, cfg.kv_heads, L.k_norm_ctx, lanes[j].position - own);
                    vc[j] = try d.heads(&lg, vj, cfg.kv_heads, null, null);
                    off += own;
                }
            }
            for (lanes, 0..) |lane, j| {
                if (full_read[j]) continue;
                const n = n_taps[j];
                const own = own_rows[j];
                if (own < n) {
                    const cc = lane.cache.?;
                    if (own > 0) {
                        kc[j] = try lg.concat(&.{ cc.k[i], kc[j] }, 2);
                        vc[j] = try lg.concat(&.{ cc.v[i], vc[j] }, 2);
                    } else {
                        kc[j] = cc.k[i];
                        vc[j] = cc.v[i];
                    }
                    const total = ops.shape(kc[j])[2];
                    kc[j] = try lg.sliceAxis(kc[j], 2, total - n, total);
                    vc[j] = try lg.sliceAxis(vc[j], 2, total - n, total);
                }
                if (lane.cache) |cc| {
                    const k_held = lg.escape(kc[j]);
                    const v_held = lg.escape(vc[j]);
                    if (cc.k[i].ctx != null) _ = mlx.mlx_array_free(cc.k[i]);
                    if (cc.v[i].ctx != null) _ = mlx.mlx_array_free(cc.v[i]);
                    cc.k[i] = k_held;
                    cc.v[i] = v_held;
                }
            }
            const x = try lg.rmsNorm(h, L.in_norm, cfg.eps);
            var q: Array = undefined;
            var kb: Array = undefined;
            var vb: Array = undefined;
            if (L.qkv.w.ctx != null) {
                const qkv = try project(&lg, &L.qkv, x);
                q = try d.headsRope(&lg, try lg.sliceAxis(qkv, 2, 0, q_dim), cfg.heads, L.q_norm, rope);
                kb = try d.headsRope(&lg, try lg.sliceAxis(qkv, 2, q_dim, q_dim + kv_dim), cfg.kv_heads, L.k_norm, rope);
                vb = try d.headsRope(&lg, try lg.sliceAxis(qkv, 2, q_dim + kv_dim, q_dim + 2 * kv_dim), cfg.kv_heads, null, .none);
            } else {
                q = try d.headsRope(&lg, try project(&lg, &L.q, x), cfg.heads, L.q_norm, rope);
                kb = try d.headsRope(&lg, try project(&lg, &L.k, x), cfg.kv_heads, L.k_norm, rope);
                vb = try d.headsRope(&lg, try project(&lg, &L.v, x), cfg.kv_heads, null, .none);
            }
            var reads: [max_lanes]Array = undefined;
            for (lanes, 0..) |lane, j| {
                const at_j: c_int = @intCast(j);
                const solo = count == 1;
                const qj = if (solo) q else try lg.sliceAxis(q, 0, at_j, at_j + 1);
                const kbj = if (solo) kb else try lg.sliceAxis(kb, 0, at_j, at_j + 1);
                const vbj = if (solo) vb else try lg.sliceAxis(vb, 0, at_j, at_j + 1);
                var keys: Array = .{};
                var values: Array = .{};
                if (full_read[j]) {
                    const cc = lane.cache.?;
                    const n_full: c_int = @intCast(cc.full_end - cc.full_from);
                    const sh = ops.shape(cc.fk[i]);
                    const end = n_full + block;
                    if (end <= sh[2]) {
                        for ([2]Array{ kbj, vbj }, [2]*Array{ &cc.fk[i], &cc.fv[i] }, [2]*Array{ &keys, &values }) |x_new, dst, out_kv| {
                            const next = lg.escape(try lg.sliceUpdate(dst.*, try lg.astype(x_new, ops.dtype(dst.*)), &.{ 0, 0, n_full, 0 }, &.{ 1, sh[1], end, sh[3] }));
                            _ = mlx.mlx_array_free(dst.*);
                            dst.* = next;
                            out_kv.* = try lg.sliceAxis(next, 2, 0, end);
                        }
                    }
                }
                if (keys.ctx == null) {
                    keys = try lg.concat(&.{ kc[j], kbj }, 2);
                    values = try lg.concat(&.{ vc[j], vbj }, 2);
                }
                reads[j] = try lg.sdpa(qj, keys, values, scale, .none);
            }
            const attn = if (count == 1) reads[0] else try lg.concat(reads[0..count], 0);
            const read = try lg.reshape(try lg.transpose(attn, &back), &.{ @intCast(count), block, -1 });
            const mid = try lg.add(h, try project(&lg, &L.o, read));
            const y = try lg.rmsNorm(mid, L.post_norm, cfg.eps);
            const mlp = if (L.gate_up.w.ctx != null) blk: {
                const gu = try project(&lg, &L.gate_up, y);
                break :blk try project(&lg, &L.down, try lg.mul(try lg.silu(try lg.sliceAxis(gu, 2, 0, inter_dim)), try lg.sliceAxis(gu, 2, inter_dim, 2 * inter_dim)));
            } else try project(&lg, &L.down, try lg.mul(try lg.silu(try project(&lg, &L.gate, y)), try project(&lg, &L.up, y)));
            h = try g.keep(try lg.add(mid, mlp));
        }

        for (lanes) |lane| if (lane.cache) |cc| {
            cc.end = @intCast(lane.position);
        };
        const rows = try g.sliceAxis(try g.rmsNorm(h, d.norm, cfg.eps), 1, 1, 1 + drafts);
        const flat_rows = try g.reshape(rows, &.{ 1, @as(c_int, @intCast(count)) * drafts, hidden });
        const base_flat = try g.astype(try project(g, if (d.head) |*own| own else head, flat_rows), .float32);
        const base = try g.reshape(base_flat, &.{ @intCast(count), drafts, -1 });

        const newest_in = if (count == 1) last_taps[0] else try g.concat(last_taps[0..count], 0);
        const newest = try d.mix(g, newest_in, @intCast(d.layers.len), d.rnn_norm);
        const t1s = if (count == 1) t1_parts[0] else try g.concat(t1_parts[0..count], 0);
        return .{ .rows = rows, .newest = newest, .base = base, .t1s = t1s, .drafts = drafts };
    }

    /// Batched draft across `lanes.len` (1..16) slots: shares all 3 drafter layer projections, `lm_head` and the
    /// Markov `fold` read across lanes while each lane attends its own context KV.
    pub fn draftBatchPacked(d: *const Drafter, g: *Graph, lanes: []const DraftLane, head: *const Linear, vocab: c_int, softcap: f32, draws: ?Draws) !BatchBlock {
        const b = try d.draftBatchBase(g, lanes, head);
        return d.markovBatch(g, b.t1s, b.rows, b.newest, b.base, vocab, softcap, kernels.markovRuns(g, d.rnn.w, d.fold, @intCast(d.cfg.rank)), draws);
    }

    /// GPU-fused batched sampled/mixed draft across `lanes.len` (1..16) slots: shares one `fold` read per tile and
    /// emits sparse top-K (`K = markov_sample_k`) proposal tables filtered to match each lane's `SampleLane` params.
    pub fn draftBatchSampled(d: *const Drafter, g: *Graph, lanes: []const DraftLane, sample_lanes: []const kernels.SampleLane, seeds: []const u64, head: *const Linear, vocab: c_int, softcap: f32) !SampledBlock {
        const b = try d.draftBatchBase(g, lanes, head);
        const rank: c_int = @intCast(d.cfg.rank);
        const count: c_int = @intCast(lanes.len);
        const steps: usize = @intCast(b.drafts);
        const dt = ops.dtype(d.norm);
        var state = try g.zeros(&.{ count, 1, rank }, dt);
        var prev = b.t1s;
        const dense_rows = try g.contiguous(b.rows);
        const dense_new = try g.contiguous(b.newest);
        const dense_base = try g.contiguous(b.base);
        const cap = try g.fromSlice(f32, &.{softcap}, &.{1});
        var lp_buf: [max_lanes * 4]f32 = undefined;
        var seed_buf: [max_lanes * 2]u32 = undefined;
        for (sample_lanes[0..lanes.len], seeds[0..lanes.len], 0..) |sl, sd, i| {
            lp_buf[i * 4 + 0] = sl.temperature;
            lp_buf[i * 4 + 1] = @floatFromInt(sl.top_k);
            lp_buf[i * 4 + 2] = sl.top_p;
            lp_buf[i * 4 + 3] = sl.min_p;
            seed_buf[i * 2 + 0] = @truncate(sd);
            seed_buf[i * 2 + 1] = @truncate(sd >> 32);
        }
        const lparams = try g.fromSlice(f32, lp_buf[0 .. lanes.len * 4], &.{ count, 4 });
        const seeds_arr = try g.fromSlice(u32, seed_buf[0 .. lanes.len * 2], &.{ count, 2 });
        var toks: [abi.max_round_tokens]Array = undefined;
        var cands: [abi.max_round_tokens]Array = undefined;
        var probs: [abi.max_round_tokens]Array = undefined;
        var draws: [abi.max_round_tokens]Array = undefined;
        for (0..steps) |i| {
            const st = try kernels.markovStepBatchSampled(g, d.rnn.w, d.rnn_bias, state, d.prev_rows, dense_rows, dense_new, prev, d.fold, d.fold_bias, dense_base, @intCast(i), vocab, cap, lparams, seeds_arr);
            state = st.state;
            prev = st.token;
            toks[i] = try g.expandDims(st.token, 1);
            cands[i] = try g.expandDims(st.cands, 1);
            probs[i] = try g.expandDims(st.probs, 1);
            draws[i] = try g.expandDims(st.draw, 1);
        }
        return .{
            .tokens = try g.concat(toks[0..steps], 1),
            .cands = try g.concat(cands[0..steps], 1),
            .probs = try g.concat(probs[0..steps], 1),
            .draws = try g.concat(draws[0..steps], 1),
            .k = kernels.markov_sample_k,
        };
    }

    pub fn draftBatch(d: *const Drafter, g: *Graph, lanes: []const DraftLane, head: *const Linear, vocab: c_int, softcap: f32, draws: ?Draws, out: []Block) !void {
        const packed_b = try d.draftBatchPacked(g, lanes, head, vocab, softcap, draws);
        for (lanes, out[0..lanes.len], 0..) |lane, *dst, j| {
            const at: c_int = @intCast(j);
            const steps: c_int = @intCast(@min(lane.steps, d.cfg.block - 1));
            const tok_row = try g.reshape(try g.sliceAxis(packed_b.tokens, 0, at, at + 1), &.{ops.shape(packed_b.tokens)[1]});
            const conf_row = try g.reshape(try g.sliceAxis(packed_b.confidence, 0, at, at + 1), &.{ops.shape(packed_b.confidence)[1]});
            dst.* = .{
                .tokens = if (ops.shape(tok_row)[0] == steps) tok_row else try g.sliceAxis(tok_row, 0, 0, steps),
                .confidence = if (ops.shape(conf_row)[0] == steps) conf_row else try g.sliceAxis(conf_row, 0, 0, steps),
                .probs = if (packed_b.probs.ctx != null) blk: {
                    const pr = try g.reshape(try g.sliceAxis(packed_b.probs, 0, at, at + 1), &.{ ops.shape(packed_b.probs)[1], vocab });
                    break :blk if (ops.shape(pr)[0] == steps) pr else try g.sliceAxis(pr, 0, 0, steps);
                } else .{},
            };
        }
    }

    /// Batched Markov chain over `B` lanes: `t1` `[B]` int32, `rows` `[B, drafts, hidden]`, `newest` `[B, 1, hidden]`,
    /// `base` `[B, drafts, V_pad]` f32.
    pub fn markovBatch(d: *const Drafter, g: *Graph, t1: Array, rows: Array, newest: Array, base: Array, vocab: c_int, softcap: f32, fused: bool, draws: ?Draws) !BatchBlock {
        const rank: c_int = @intCast(d.cfg.rank);
        const count = ops.shape(rows)[0];
        const steps: usize = @intCast(ops.shape(rows)[1]);
        const dt = ops.dtype(d.norm);
        var state = try g.zeros(&.{ count, 1, rank }, dt);
        var prev = t1;
        var drafted: [abi.max_round_tokens]Array = undefined;
        var sure: [abi.max_round_tokens]Array = undefined;
        var probs: [abi.max_round_tokens]Array = undefined;
        const dense_rows = try g.contiguous(rows);
        const dense_new = try g.contiguous(newest);
        const dense_base = try g.contiguous(base);
        var inv_temp: Array = .{};
        if (draws) |dr| {
            var inv: [max_lanes]f32 = undefined;
            for (dr.temperatures, inv[0..dr.temperatures.len]) |temp, *dst| dst.* = 1.0 / temp;
            inv_temp = try g.fromSlice(f32, inv[0..dr.temperatures.len], &.{ count, 1, 1 });
        }
        for (0..steps) |i| {
            const at: c_int = @intCast(i);
            if (fused and draws == null) {
                const st = try kernels.markovStepBatch(g, d.rnn.w, d.rnn_bias, state, d.prev_rows, dense_rows, dense_new, prev, d.fold, d.fold_bias, dense_base, at, vocab, softcap);
                state = st.state;
                prev = st.token;
                drafted[i] = try g.expandDims(st.token, 1);
                sure[i] = try g.expandDims(st.confidence, 1);
                continue;
            }
            const last = try g.astype(try g.reshape(try g.take(d.prev_rows, prev, 0), &.{ count, 1, rank }), dt);
            const fed = try g.concat(&.{ state, last, try g.sliceAxis(rows, 1, at, at + 1), newest }, 2);
            const p = try g.astype(try g.add(try project(g, &d.rnn, fed), d.rnn_bias), .float32);
            const keep = try g.astype(try g.sigmoid(try g.sliceAxis(p, 2, 0, rank)), dt);
            const blend = try g.mul(try g.sub(try g.scalar(1, dt), keep), try g.astype(try g.tanh(try g.sliceAxis(p, 2, rank, 2 * rank)), dt));
            state = try g.add(try g.mul(keep, state), blend);
            const latent = try g.astype(try g.tanh(try g.sliceAxis(p, 2, 2 * rank, 3 * rank)), dt);
            const bias = try g.astype(try g.add(try g.matmul(latent, d.fold), d.fold_bias), .float32);
            const logits = try g.sliceAxis(try g.add(try g.sliceAxis(base, 1, at, at + 1), bias), 2, 0, vocab);
            const capped = if (softcap > 0) try g.mulScalar(try g.tanh(try g.mulScalar(logits, 1.0 / softcap)), softcap) else logits;
            if (draws) |dr| {
                const scaled = try g.mul(capped, inv_temp);
                const key = try g.call(mlx.mlx_random_key, .{dr.seed +% @as(u64, i) *% 0x9E3779B97F4A7C15});
                prev = try g.astype(try g.reshape(try g.call(mlx.mlx_random_categorical, .{ scaled, @as(c_int, -1), key }), &.{count}), .int32);
                const pr = try g.softmax(scaled, -1, true);
                probs[i] = pr;
                sure[i] = try g.reshape(try g.max(pr, -1, false), &.{ count, 1 });
            } else {
                prev = try g.astype(try g.reshape(try g.argmax(logits, -1, false), &.{count}), .int32);
                sure[i] = try g.reshape(try g.exp(try g.sub(try g.max(capped, -1, false), try g.call(mlx.mlx_logsumexp_axis, .{ capped, -1, false }))), &.{ count, 1 });
            }
            drafted[i] = try g.expandDims(prev, 1);
        }
        return .{
            .tokens = try g.concat(drafted[0..steps], 1),
            .confidence = try g.concat(sure[0..steps], 1),
            .probs = if (draws != null) try g.concat(probs[0..steps], 1) else .{},
        };
    }

    /// The `block - 1` tokens drafted after the committed token `t1` (`[1]` int32) at position `position`. `anchor` `[1, 1, hidden]` is `t1`'s normed embedding, `taps` `[1, n, taps * hidden]` the trunk's
    /// taps at the `n >= 1` positions before `position`, `head` the trunk's LM head, `vocab` its real rows and `softcap` its
    /// logit soft cap. `cache` carries the context's keys and values over the slot's rounds (null: projected afresh).
    pub fn draft(d: *const Drafter, g: *Graph, t1: Array, anchor: Array, taps: Array, position: c_int, head: *const Linear, vocab: c_int, softcap: f32, cache: ?*ContextKv, law: ?Law) !Block {
        const cfg = &d.cfg;
        const hidden: c_int = @intCast(d.hidden);
        const block: c_int = @intCast(cfg.block);
        const rank: c_int = @intCast(cfg.rank);
        const n = ops.shape(taps)[1];
        const dt = ops.dtype(d.norm);
        const at: u64 = @intCast(position);
        // The rows the cache lacks: those committed since its end, when it still reaches back to the ring's first.
        var fresh: c_int = n;
        if (cache) |cc| if (cc.k[0].ctx != null and cc.end <= at and at - cc.end <= @as(u64, @intCast(n))) {
            const since: c_int = @intCast(at - cc.end);
            if (since + ops.shape(cc.k[0])[2] >= n) fresh = since;
        };
        const tapsOf = struct {
            fn f(gr: *Graph, x: Array, from: c_int, to: c_int, count: u32, width: c_int) !Array {
                return gr.astype(try gr.reshape(try gr.sliceAxis(x, 1, from, to), &.{ 1, to - from, @intCast(count), width }), .float32);
            }
        }.f;
        const new32 = if (fresh > 0) try tapsOf(g, taps, n - fresh, n, cfg.tap_count, hidden) else null;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const back = [_]c_int{ 0, 2, 1, 3 };

        var h = try g.concat(&.{ try g.astype(anchor, dt), try g.broadcast(try g.astype(d.mask, dt), &.{ 1, block - 1, hidden }) }, 1);
        for (d.layers, 0..) |*L, i| {
            var lg = g.scope();
            defer lg.deinit();
            var kc: Array = .{};
            var vc: Array = .{};
            // A full layer reads every committed position (kept by `appendFull`), not the ring's newest.
            if (cache) |cc| if (cfg.full(i) and cc.fk[i].ctx != null and cc.full_end == at) {
                const n_full: c_int = @intCast(cc.full_end - cc.full_from);
                kc = try lg.sliceAxis(cc.fk[i], 2, 0, n_full);
                vc = try lg.sliceAxis(cc.fv[i], 2, 0, n_full);
            };
            const full_read = kc.ctx != null;
            // A layer with no ring copy (a full one whose buffers end elsewhere) projects the whole ring.
            const own: c_int = if (fresh < n and cache.?.k[i].ctx == null) n else fresh;
            if (full_read) {} else if (own > 0) {
                const rows32 = if (own == fresh) new32.? else try tapsOf(&lg, taps, 0, n, cfg.tap_count, hidden);
                const context = try d.mix(&lg, rows32, @intCast(i), L.mix_norm);
                kc = try d.heads(&lg, try project(&lg, &L.k_ctx, context), cfg.kv_heads, L.k_norm_ctx, position - own);
                vc = try d.heads(&lg, try project(&lg, &L.v_ctx, context), cfg.kv_heads, null, null);
            }
            if (!full_read and own < n) {
                const cc = cache.?;
                if (own > 0) {
                    kc = try lg.concat(&.{ cc.k[i], kc }, 2);
                    vc = try lg.concat(&.{ cc.v[i], vc }, 2);
                } else {
                    kc = cc.k[i];
                    vc = cc.v[i];
                }
                const total = ops.shape(kc)[2];
                kc = try lg.sliceAxis(kc, 2, total - n, total);
                vc = try lg.sliceAxis(vc, 2, total - n, total);
            }
            if (cache) |cc| if (!full_read) {
                const k_held = lg.escape(kc);
                const v_held = lg.escape(vc);
                if (cc.k[i].ctx != null) _ = mlx.mlx_array_free(cc.k[i]);
                if (cc.v[i].ctx != null) _ = mlx.mlx_array_free(cc.v[i]);
                cc.k[i] = k_held;
                cc.v[i] = v_held;
            };
            const x = try lg.rmsNorm(h, L.in_norm, cfg.eps);
            const q = try d.heads(&lg, try project(&lg, &L.q, x), cfg.heads, L.q_norm, position);
            const kb = try d.heads(&lg, try project(&lg, &L.k, x), cfg.kv_heads, L.k_norm, position);
            const vb = try d.heads(&lg, try project(&lg, &L.v, x), cfg.kv_heads, null, null);
            var keys: Array = .{};
            var values: Array = .{};
            if (full_read) {
                // The block's keys go in place after the context, as one view: no copy of the whole context. The next
                // push overwrites them.
                const cc = cache.?;
                const n_full: c_int = @intCast(cc.full_end - cc.full_from);
                const sh = ops.shape(cc.fk[i]);
                const end = n_full + block;
                if (end <= sh[2]) {
                    for ([2]Array{ kb, vb }, [2]*Array{ &cc.fk[i], &cc.fv[i] }, [2]*Array{ &keys, &values }) |x_new, dst, out| {
                        const next = lg.escape(try lg.sliceUpdate(dst.*, try lg.astype(x_new, ops.dtype(dst.*)), &.{ 0, 0, n_full, 0 }, &.{ 1, sh[1], end, sh[3] }));
                        _ = mlx.mlx_array_free(dst.*);
                        dst.* = next;
                        out.* = try lg.sliceAxis(next, 2, 0, end);
                    }
                }
            }
            if (keys.ctx == null) {
                keys = try lg.concat(&.{ kc, kb }, 2);
                values = try lg.concat(&.{ vc, vb }, 2);
            }
            const read = try lg.reshape(try lg.transpose(try lg.sdpa(q, keys, values, scale, .none), &back), &.{ 1, block, -1 });
            const mid = try lg.add(h, try project(&lg, &L.o, read));
            const y = try lg.rmsNorm(mid, L.post_norm, cfg.eps);
            const mlp = try project(&lg, &L.down, try lg.mul(try lg.silu(try project(&lg, &L.gate, y)), try project(&lg, &L.up, y)));
            h = try g.keep(try lg.add(mid, mlp));
        }
        const drafts: c_int = if (law) |l| @min(block - 1, @as(c_int, @intCast(@min(l.steps, block - 1)))) else block - 1;
        const rows = try g.sliceAxis(try g.rmsNorm(h, d.norm, cfg.eps), 1, 1, 1 + drafts);
        const base = try g.astype(try project(g, if (d.head) |*own| own else head, rows), .float32);

        if (cache) |cc| cc.end = at;
        // The RNN restarts each block from the newest committed position's mix.
        const newest = try d.mix(g, try tapsOf(g, taps, n - 1, n, cfg.tap_count, hidden), @intCast(d.layers.len), d.rnn_norm);
        return d.markov(g, t1, rows, newest, base, vocab, softcap, kernels.markovRuns(g, d.rnn.w, d.fold, rank), law);
    }

    /// The Markov head's chain over the block rows `rows` `[1, block - 1, hidden]` and their head logits `base` f32,
    /// from the committed token `t1` and the newest mix `newest` `[1, 1, hidden]`: each step's token and probability.
    /// `fused` runs an argmax step as `kernels.markovStep`'s three launches, else as the op chain; a `law` draws each step
    /// on the op chain, which keeps the whole row.
    pub fn markov(d: *const Drafter, g: *Graph, t1: Array, rows: Array, newest: Array, base: Array, vocab: c_int, softcap: f32, fused: bool, law: ?Law) !Block {
        const rank: c_int = @intCast(d.cfg.rank);
        const steps: usize = @intCast(ops.shape(rows)[1]);
        const dt = ops.dtype(d.norm);
        var state = try g.zeros(&.{ 1, 1, rank }, dt);
        var prev = t1;
        var drafted: [abi.max_round_tokens]Array = undefined;
        var sure: [abi.max_round_tokens]Array = undefined;
        var probs: [abi.max_round_tokens]Array = undefined;
        const dense_rows = try g.contiguous(rows);
        for (0..steps) |i| {
            const at: c_int = @intCast(i);
            if (fused and law == null) {
                const st = try kernels.markovStep(g, d.rnn.w, d.rnn_bias, state, d.prev_rows, dense_rows, newest, prev, d.fold, d.fold_bias, base, at, vocab, softcap);
                state = st.state;
                prev = st.token;
                drafted[i] = st.token;
                sure[i] = st.confidence;
                continue;
            }
            const last = try g.astype(try g.reshape(try g.take(d.prev_rows, prev, 0), &.{ 1, 1, rank }), dt);
            const fed = try g.concat(&.{ state, last, try g.sliceAxis(rows, 1, at, at + 1), newest }, 2);
            const p = try g.astype(try g.add(try project(g, &d.rnn, fed), d.rnn_bias), .float32);
            const keep = try g.astype(try g.sigmoid(try g.sliceAxis(p, 2, 0, rank)), dt);
            const blend = try g.mul(try g.sub(try g.scalar(1, dt), keep), try g.astype(try g.tanh(try g.sliceAxis(p, 2, rank, 2 * rank)), dt));
            state = try g.add(try g.mul(keep, state), blend);
            const latent = try g.astype(try g.tanh(try g.sliceAxis(p, 2, 2 * rank, 3 * rank)), dt);
            const bias = try g.astype(try g.add(try g.matmul(latent, d.fold), d.fold_bias), .float32);
            // The soft cap is monotonic: the pick over the real rows is the capped logits'; the probability is not.
            const logits = try g.sliceAxis(try g.add(try g.sliceAxis(base, 1, at, at + 1), bias), 2, 0, vocab);
            const capped = if (softcap > 0) try g.mulScalar(try g.tanh(try g.mulScalar(logits, 1.0 / softcap)), softcap) else logits;
            if (law) |l| {
                const scaled = try g.mulScalar(capped, 1.0 / l.temperature);
                const key = try g.call(mlx.mlx_random_key, .{l.seed +% @as(u64, i) *% 0x9E3779B97F4A7C15});
                prev = try g.astype(try g.reshape(try g.call(mlx.mlx_random_categorical, .{ scaled, @as(c_int, -1), key }), &.{1}), .int32);
                probs[i] = try g.reshape(try g.softmax(scaled, -1, true), &.{ 1, vocab });
                // How sure the law is (its peak), the argmax step's measure: a drawn token's own probability is not.
                sure[i] = try g.reshape(try g.max(probs[i], -1, false), &.{1});
            } else {
                prev = try g.astype(try g.reshape(try g.argmax(logits, -1, false), &.{1}), .int32);
                sure[i] = try g.reshape(try g.exp(try g.sub(try g.max(capped, -1, false), try g.call(mlx.mlx_logsumexp_axis, .{ capped, -1, false }))), &.{1});
            }
            drafted[i] = prev;
        }
        return .{
            .tokens = try g.concat(drafted[0..steps], 0),
            .confidence = try g.concat(sure[0..steps], 0),
            .probs = if (law != null) try g.concat(probs[0..steps], 0) else .{},
        };
    }
};

/// A dense `head` as the affine 8-bit weight the drafter reads (owned by `g`); null for a quantized one.
pub fn headCopy(g: *Graph, head: *const Linear) !?Linear {
    if (head.s.ctx != null) return null;
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, head.w, mlx.mlx_optional_int.some(draft_head_group), mlx.mlx_optional_int.some(draft_head_bits), "affine", .{}, g.s));
    var own: Linear = .{ .bits = draft_head_bits, .group_size = draft_head_group };
    inline for (.{ &own.w, &own.s, &own.b }, 0..) |dst, i| {
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&x, parts, i));
        dst.* = try g.adopt(x);
    }
    return own;
}

/// Reads the sidecar's tensors (a layer's under `layers.N.`), owned by the drafter.
pub const At = struct {
    owned: *std.ArrayList(Array),
    w: *const Weights,
    g: *Graph,
    layer: ?u32 = null,

    /// A new handle the graph owns.
    pub fn raw(a: At, leaf: []const u8) !Array {
        var buf: [160]u8 = undefined;
        const name = if (a.layer) |i| try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, leaf }) else leaf;
        const x = a.w.get(name) orelse return error.MissingWeight;
        return a.g.keep(x);
    }

    pub fn keep(a: At, x: Array) !Array {
        try ops.eval(x);
        const out = a.g.escape(x);
        try a.owned.append(gpa, out);
        return out;
    }

    pub fn need(a: At, leaf: []const u8) !Array {
        return a.keep(try a.raw(leaf));
    }

    pub fn float(a: At, leaf: []const u8) !Array {
        return a.keep(try a.g.astype(try a.raw(leaf), .float32));
    }

    /// The head the block rows are read through: a dense trunk head as an affine 8-bit copy in groups of 64, owned
    /// by the drafter; null for a quantized one, read as it is. The rows are drafts (the trunk verifies each one),
    /// and the head is a round's largest read: at 8 bits it is half the bytes.
    pub fn draftHead(a: At, head: *const Linear) !?Linear {
        var own = (try headCopy(a.g, head)) orelse return null;
        for ([_]*Array{ &own.w, &own.s, &own.b }) |x| x.* = try a.keep(x.*);
        return own;
    }

    /// A projection in the checkpoint's layout, `[out, in]`.
    pub fn linear(a: At, base: []const u8) !Linear {
        var buf: [160]u8 = undefined;
        return .{ .w = try a.need(try std.fmt.bufPrint(&buf, "{s}.weight", .{base})) };
    }
};
