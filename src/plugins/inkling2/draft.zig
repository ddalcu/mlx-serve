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
    q: Linear = .{},
    k: Linear = .{},
    v: Linear = .{},
    qkv: Linear = .{},
    o: Linear,
    k_ctx: Linear = .{},
    v_ctx: Linear = .{},
    kv_ctx: Linear = .{},
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    gate: Linear = .{},
    up: Linear = .{},
    gate_up: Linear = .{},
    down: Linear,
};

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
    /// Group-64 affine 8-bit copy of `fold` (`[rank, V]` uint8 + `[rank/64, V]` f16 scale/bias) so the 7-step Markov
    /// chain fits in SLC (~55 MB vs 103 MB). `fold` stays for the CPU/op-chain fallback.
    fold_q: Array = .{},
    fold_s: Array = .{},
    fold_qb: Array = .{},
    /// An 8-bit copy of a dense trunk head, which the block rows are read through (`draftHead`); null = the trunk's.
    head: ?Linear = null,

    /// The drafter beside the pack, its Markov head folded over the trunk's `head`; null when the pack ships none.
    /// `bits` quantizes the layer projections (affine g64) and `fold` (group-64 uint8); 0 keeps BF16 as shipped.
    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Draft, hidden: u32, head: *const Linear, s: mlx.mlx_stream, bits: u32) !?*Drafter {
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
                .qkv = try at.narrowedJoined(&.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, bits),
                .o = try at.narrowed("self_attn.o_proj", bits),
                .kv_ctx = try at.narrowedJoined(&.{ "self_attn.k_proj_ctx", "self_attn.v_proj_ctx" }, bits),
                .q_norm = try at.need("self_attn.q_norm.weight"),
                .k_norm = try at.need("self_attn.k_norm.weight"),
                .k_norm_ctx = try at.need("self_attn.k_norm_ctx.weight"),
                .gate_up = try at.narrowedJoined(&.{ "mlp.gate_proj", "mlp.up_proj" }, bits),
                .down = try at.narrowed("mlp.down_proj", bits),
            };
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
        if (bits > 0 and @rem(rank, 64) == 0 and @rem(ops.shape(d.fold)[1], 8) == 0 and (ops.dtype(d.fold) == .bfloat16 or ops.dtype(d.fold) == .float16)) {
            const qf = try quantizeFold(&g, d.fold, 64);
            d.fold_q = try at.keep(qf[0]);
            d.fold_s = try at.keep(qf[1]);
            d.fold_qb = try at.keep(qf[2]);
        }
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
            const kv_dim: c_int = @intCast(d.cfg.kv_heads * d.cfg.head_dim);
            var r0: c_int = 0;
            while (r0 < n) : (r0 += 1024) {
                const r1 = @min(n, r0 + 1024);
                const rows32 = try lg.astype(try lg.reshape(try sideBySide(&lg, parts, r0, r1), &.{ 1, r1 - r0, @intCast(d.cfg.tap_count), @intCast(d.hidden) }), .float32);
                const context = try d.mix(&lg, rows32, @intCast(i), L.mix_norm);
                const k, const v = if (L.kv_ctx.w.ctx != null) blk: {
                    const kv_c = try project(&lg, &L.kv_ctx, context);
                    break :blk .{
                        try d.heads(&lg, try lg.sliceAxis(kv_c, 2, 0, kv_dim), d.cfg.kv_heads, L.k_norm_ctx, @as(c_int, @intCast(first)) + r0),
                        try d.heads(&lg, try lg.sliceAxis(kv_c, 2, kv_dim, 2 * kv_dim), d.cfg.kv_heads, null, null),
                    };
                } else .{
                    try d.heads(&lg, try project(&lg, &L.k_ctx, context), d.cfg.kv_heads, L.k_norm_ctx, @as(c_int, @intCast(first)) + r0),
                    try d.heads(&lg, try project(&lg, &L.v_ctx, context), d.cfg.kv_heads, null, null),
                };
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

        const q_dim: c_int = @intCast(cfg.heads * cfg.head_dim);
        const kv_dim: c_int = @intCast(cfg.kv_heads * cfg.head_dim);
        const inter_dim: c_int = @intCast(cfg.intermediate);
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
                if (L.kv_ctx.w.ctx != null) {
                    const kv_c = try project(&lg, &L.kv_ctx, context);
                    kc = try d.heads(&lg, try lg.sliceAxis(kv_c, 2, 0, kv_dim), cfg.kv_heads, L.k_norm_ctx, position - own);
                    vc = try d.heads(&lg, try lg.sliceAxis(kv_c, 2, kv_dim, 2 * kv_dim), cfg.kv_heads, null, null);
                } else {
                    kc = try d.heads(&lg, try project(&lg, &L.k_ctx, context), cfg.kv_heads, L.k_norm_ctx, position - own);
                    vc = try d.heads(&lg, try project(&lg, &L.v_ctx, context), cfg.kv_heads, null, null);
                }
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
            const q, const kb, const vb = if (L.qkv.w.ctx != null) blk: {
                const qkv = try project(&lg, &L.qkv, x);
                break :blk .{
                    try d.heads(&lg, try lg.sliceAxis(qkv, 2, 0, q_dim), cfg.heads, L.q_norm, position),
                    try d.heads(&lg, try lg.sliceAxis(qkv, 2, q_dim, q_dim + kv_dim), cfg.kv_heads, L.k_norm, position),
                    try d.heads(&lg, try lg.sliceAxis(qkv, 2, q_dim + kv_dim, q_dim + 2 * kv_dim), cfg.kv_heads, null, null),
                };
            } else .{
                try d.heads(&lg, try project(&lg, &L.q, x), cfg.heads, L.q_norm, position),
                try d.heads(&lg, try project(&lg, &L.k, x), cfg.kv_heads, L.k_norm, position),
                try d.heads(&lg, try project(&lg, &L.v, x), cfg.kv_heads, null, null),
            };
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
            const gated_mlp = if (L.gate_up.w.ctx != null) blk: {
                const gu = try project(&lg, &L.gate_up, y);
                break :blk try lg.mul(try lg.silu(try lg.sliceAxis(gu, 2, 0, inter_dim)), try lg.sliceAxis(gu, 2, inter_dim, 2 * inter_dim));
            } else try lg.mul(try lg.silu(try project(&lg, &L.gate, y)), try project(&lg, &L.up, y));
            const mlp = try project(&lg, &L.down, gated_mlp);
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
        if (fused and steps > 0) {
            const cap = try g.fromSlice(f32, &.{softcap}, &.{1});
            const q8 = d.fold_q.ctx != null;
            if (law) |l| if (q8) {
                const inv_t: u32 = @bitCast(@as(f32, 1.0 / l.temperature));
                const lparams = try g.fromSlice(u32, &.{ inv_t, @truncate(l.seed), @truncate(l.seed >> 32), 0 }, &.{4});
                var scaled_rows: [abi.max_round_tokens]Array = undefined;
                var lse_rows: [abi.max_round_tokens]Array = undefined;
                const step0 = try g.fromSlice(i32, &.{0}, &.{1});
                var p = try kernels.markovRnn(g, d.rnn.w, d.rnn_bias, state, d.prev_rows, dense_rows, newest, prev, step0);
                var f = try kernels.markovFoldQ8Law(g, p, d.fold_q, d.fold_s, d.fold_qb, d.fold_bias, base, step0, vocab, cap, lparams, dt);
                scaled_rows[0] = f[4];
                for (1..steps) |i| {
                    const step = try g.fromSlice(i32, &.{@intCast(i)}, &.{1});
                    const pr = try kernels.markovPickRnnLaw(g, .{ f[0], f[1], f[2], f[3] }, p, state, cap, d.rnn.w, d.rnn_bias, d.prev_rows, dense_rows, newest, step, true);
                    drafted[i - 1] = pr[0];
                    sure[i - 1] = pr[1];
                    state = pr[2];
                    p = pr[3];
                    lse_rows[i - 1] = pr[4];
                    f = try kernels.markovFoldQ8Law(g, p, d.fold_q, d.fold_s, d.fold_qb, d.fold_bias, base, step, vocab, cap, lparams, dt);
                    scaled_rows[i] = f[4];
                }
                const last = try kernels.markovPickRnnLaw(g, .{ f[0], f[1], f[2], f[3] }, p, state, cap, d.rnn.w, d.rnn_bias, d.prev_rows, dense_rows, newest, step0, false);
                drafted[steps - 1] = last[0];
                sure[steps - 1] = last[1];
                lse_rows[steps - 1] = last[4];
                const scaled_all = if (steps == 1) scaled_rows[0] else try g.concat(scaled_rows[0..steps], 0);
                const lse_all = try g.reshape(if (steps == 1) lse_rows[0] else try g.concat(lse_rows[0..steps], 0), &.{ @intCast(steps), 1 });
                return .{
                    .tokens = try g.concat(drafted[0..steps], 0),
                    .confidence = try g.concat(sure[0..steps], 0),
                    .probs = try g.exp(try g.sub(scaled_all, lse_all)),
                };
            };
            if (law == null) {
                const step0 = try g.fromSlice(i32, &.{0}, &.{1});
                var p = try kernels.markovRnn(g, d.rnn.w, d.rnn_bias, state, d.prev_rows, dense_rows, newest, prev, step0);
                var part = if (q8)
                    try kernels.markovFoldQ8(g, p, d.fold_q, d.fold_s, d.fold_qb, d.fold_bias, base, step0, vocab, cap, dt)
                else
                    try kernels.markovFold(g, p, d.fold, d.fold_bias, base, step0, vocab, cap);
                for (1..steps) |i| {
                    const step = try g.fromSlice(i32, &.{@intCast(i)}, &.{1});
                    const pr = try kernels.markovPickRnn(g, part, p, state, cap, d.rnn.w, d.rnn_bias, d.prev_rows, dense_rows, newest, step);
                    drafted[i - 1] = pr[0];
                    sure[i - 1] = pr[1];
                    state = pr[2];
                    p = pr[3];
                    part = if (q8)
                        try kernels.markovFoldQ8(g, p, d.fold_q, d.fold_s, d.fold_qb, d.fold_bias, base, step, vocab, cap, dt)
                    else
                        try kernels.markovFold(g, p, d.fold, d.fold_bias, base, step, vocab, cap);
                }
                const last = try kernels.markovPick(g, part, p, state, cap);
                drafted[steps - 1] = last[0];
                sure[steps - 1] = last[1];
                return .{
                    .tokens = try g.concat(drafted[0..steps], 0),
                    .confidence = try g.concat(sure[0..steps], 0),
                    .probs = .{},
                };
            }
        }
        for (0..steps) |i| {
            const at: c_int = @intCast(i);
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

/// Quantize `fold` `[rank, V]` along `rank` in groups of `gs` into `{ uint8 [rank, V], f16 [rank/gs, V], f16 [rank/gs, V] }`.
pub fn quantizeFold(g: *Graph, fold: Array, gs: c_int) ![3]Array {
    const sh = ops.shape(fold);
    const rank = sh[0];
    const vpad = sh[1];
    const f32_fold = try g.astype(fold, .float32);
    const f_grp = try g.reshape(f32_fold, &.{ @divExact(rank, gs), gs, vpad });
    var mn_a = mlx.mlx_array_new();
    var mx_a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_min_axis(&mn_a, f_grp, 1, false, g.s));
    try mlx.check(mlx.mlx_max_axis(&mx_a, f_grp, 1, false, g.s));
    const mn = try g.adopt(mn_a);
    const mx = try g.adopt(mx_a);
    const fs = try g.astype(try g.mulScalar(try g.sub(mx, mn), 1.0 / 255.0), .float16);
    const fqb = try g.astype(mn, .float16);
    const sc_h = try g.astype(fs, .float32);
    const mn_h = try g.astype(fqb, .float32);
    const zero_f = try g.scalar(0.0, .float32);
    const one_f = try g.scalar(1.0, .float32);
    const pos = try g.greater(sc_h, zero_f);
    const safe_sc = try g.where(pos, sc_h, one_f);
    const inv_sc = try g.where(pos, try g.div(one_f, safe_sc), zero_f);
    const scaled = try g.mul(try g.sub(f_grp, try g.expandDims(mn_h, 1)), try g.expandDims(inv_sc, 1));
    var rnd_a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_round(&rnd_a, scaled, 0, g.s));
    const rnd = try g.adopt(rnd_a);
    var clp_a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_clip(&clp_a, rnd, zero_f, try g.scalar(255.0, .float32), g.s));
    const fq = try g.reshape(try g.astype(try g.adopt(clp_a), .uint8), &.{ rank, vpad });
    return .{ fq, fs, fqb };
}

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

/// Quantize a dense matrix `w` `[out, in]` as affine `bits` in groups of 64, owned by `g`.
fn quantizeDense(g: *Graph, w: Array, bits: u32) !Linear {
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, w, mlx.mlx_optional_int.some(draft_head_group), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, g.s));
    var own: Linear = .{ .bits = @intCast(bits), .group_size = draft_head_group };
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

    /// A dense projection, quantized to affine `bits` (g64) when `bits > 0` and `in_dim % 64 == 0`.
    pub fn narrowed(a: At, base: []const u8, bits: u32) !Linear {
        var buf: [160]u8 = undefined;
        const w = try a.raw(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}));
        if (bits == 0 or @rem(ops.shape(w)[1], draft_head_group) != 0) return .{ .w = try a.keep(w) };
        var own = try quantizeDense(a.g, w, bits);
        for ([_]*Array{ &own.w, &own.s, &own.b }) |x| x.* = try a.keep(x.*);
        return own;
    }

    /// Several dense projections stacked along axis 0, quantized once to affine `bits` (g64) when eligible.
    pub fn narrowedJoined(a: At, bases: []const []const u8, bits: u32) !Linear {
        var ws: [3]Array = undefined;
        var buf: [160]u8 = undefined;
        for (bases, 0..) |base, i| ws[i] = try a.raw(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}));
        const joined = try a.g.concat(ws[0..bases.len], 0);
        if (bits == 0 or @rem(ops.shape(joined)[1], draft_head_group) != 0) return .{ .w = try a.keep(joined) };
        var own = try quantizeDense(a.g, joined, bits);
        for ([_]*Array{ &own.w, &own.s, &own.b }) |x| x.* = try a.keep(x.*);
        return own;
    }
};
