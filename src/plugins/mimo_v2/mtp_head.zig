//! MiMo-V2's MTP heads behind `SpecHooks.head` (`SpecFacts.mtp_heads`): the checkpoint's `model.mtp.layers.{k}`,
//! each one sliding-window decoder layer with sinks and a dense SwiGLU fed `eh_proj([enorm(embed(token)),
//! hnorm(trunk hidden)])` and read through its own final norm. Head k's row p pairs the trunk hidden at p with token
//! p+k+1 at rope position p: every head reads the trunk's hidden, only the token shifts. A round opens on committed
//! rows at head 0 (last row q); draft step i runs head i at q over its catch-up rows, the last i carrying the drafts.
//! A request's rows, the committed hiddens and tokens they are built from and the round's drafts ride the head
//! cache's recurrent entries (`State`); KV layer 0 only counts the committed rows (the engine's history length).
//! copied_from: plugins/mimo_v2/mtp.zig (`Head`, `State`, `RowCache`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const gatherRows = bases.blocks.embed.gatherRows;
const config = @import("config.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

pub const MAX_HEADS: usize = config.mtp_max_heads;
/// Committed hidden rows a request keeps (every head's window, a lagging head's catch-up rows, a round's drafts).
const RING_ROWS: usize = 256;
const MAX_TOKS: usize = RING_ROWS + MAX_HEADS + 1;

const Layer = struct {
    enorm: Array,
    hnorm: Array,
    input_norm: Array,
    pre_mlp_norm: Array,
    final_norm: Array,
    sinks: Array,
    eh_proj: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
};

pub const Heads = struct {
    n: usize = 0,
    layers: [MAX_HEADS]Layer = undefined,
    owned: reader.Owned = .{},
    n_heads: c_int,
    n_kv: c_int,
    head_dim: c_int,
    v_dim: c_int,
    rope_dims: c_int,
    rope_base: f32,
    value_scale: f32,
    eps: f32,
    window: usize,

    pub fn deinit(h: *Heads) void {
        h.owned.deinit();
    }
};

/// The heads the pack ships, in the sliding layers' geometry; null when it ships none.
pub fn load(ctx: *const abi.LoadCtx, c: *const config.Config) !?*Heads {
    if (c.sliding_window == 0) return error.MimoMtpNeedsSlidingGeometry;
    // MiMo's heads ride the trunk's shards; a streaming pack's ride its sidecar.
    const sidecars: []const []const u8 = if (c.streamed) &.{"mtp-heads.safetensors"} else &.{};
    var res = Resident.adopt(try reader.headWeights(ctx, sidecars, "model.mtp.layers.0.eh_proj.weight", "model.mtp."));
    defer res.deinit();
    // A Sushi pack stores them as its trunk: a fused e4m3 QKV and an e4m3 MLP.
    if (c.expert_block_json.len > 0) {
        const kv_rows: u64 = @as(u64, c.num_key_value_heads) * c.head_dim;
        try sushi_pack.adaptMimoHeads(&res.weights, .{ @as(u64, c.num_attention_heads) * c.head_dim, kv_rows, @as(u64, c.num_key_value_heads) * c.v_head_dim }, .{ .ctx = ctx.stream.ctx });
    }
    const src = res.source();
    var lctx = ctx.*;
    lctx.weights = &src;
    var r = try reader.Reader.open(&lctx, c.block_json, "model.mtp.layers");
    defer r.close();
    const h = try gpa.create(Heads);
    errdefer gpa.destroy(h);
    h.* = .{
        .n_heads = @intCast(c.num_attention_heads),
        .n_kv = @intCast(c.num_key_value_heads),
        .head_dim = @intCast(c.head_dim),
        .v_dim = @intCast(c.v_head_dim),
        .rope_dims = @intCast(c.ropeDims()),
        .rope_base = c.swa_rope_theta,
        .value_scale = c.value_scale,
        .eps = c.rms_norm_eps,
        .window = c.sliding_window,
    };
    errdefer h.deinit();
    var nb: [160]u8 = undefined;
    while (h.n < MAX_HEADS and r.has(try r.name(&nb, "{d}.eh_proj.weight", .{h.n}))) : (h.n += 1) h.layers[h.n] = try layerOf(h, &r, c, h.n);
    if (h.n == 0) {
        h.deinit();
        gpa.destroy(h);
        return null;
    }
    for (h.owned.list.items) |a| _ = mlx.mlx_array_eval(a);
    return h;
}

fn layerOf(h: *Heads, r: *reader.Reader, c: *const config.Config, k: usize) !Layer {
    var nb: [160]u8 = undefined;
    var norms: [5]Array = undefined;
    for (&norms, [_][]const u8{ "enorm", "hnorm", "input_layernorm", "pre_mlp_layernorm", "final_layernorm" }) |*x, leaf| {
        x.* = try r.need(try r.name(&nb, "{d}.{s}.weight", .{ k, leaf }));
        try h.owned.own(x.*);
    }
    const sinks = try r.share(try r.name(&nb, "{d}.self_attn.attention_sink_bias", .{k}));
    try h.owned.own(sinks);
    const hidden = c.hidden_size;
    var p: [8]Linear = undefined;
    const leaves = [_][]const u8{ "eh_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
    const ins = [_]u32{ 2 * hidden, hidden, hidden, hidden, c.num_attention_heads * c.v_head_dim, hidden, hidden, c.intermediate_size };
    for (&p, leaves, ins) |*l, leaf, in| {
        l.* = try r.proj(try r.name(&nb, "{d}.{s}", .{ k, leaf }), in);
        // The legacy head keeps a dense weight transposed AND contiguous.
        if (l.s.ctx == null) {
            var t = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_contiguous(&t, l.w, false, r.s));
            _ = mlx.mlx_array_free(l.w);
            l.w = t;
        }
        try h.owned.linear(l.*);
    }
    return .{
        .enorm = norms[0],
        .hnorm = norms[1],
        .input_norm = norms[2],
        .pre_mlp_norm = norms[3],
        .final_norm = norms[4],
        .sinks = sinks,
        .eh_proj = p[0],
        .q = p[1],
        .k = p[2],
        .v = p[3],
        .o = p[4],
        .gate = p[5],
        .up = p[6],
        .down = p[7],
    };
}

/// Rows `[base, base + len)` of one head's keys and values, the newest last.
const RowCache = struct {
    k: Array = .{},
    v: Array = .{},
    base: usize = 0,
    len: usize = 0,

    fn end(rc: *const RowCache) usize {
        return rc.base + rc.len;
    }

    fn reset(rc: *RowCache, at: usize) void {
        rc.* = .{ .base = at };
    }

    fn truncate(rc: *RowCache, g: *Graph, at: usize) !void {
        if (at >= rc.end()) return;
        if (at <= rc.base) return rc.reset(at);
        const keep: c_int = @intCast(at - rc.base);
        rc.k = try rows(g, rc.k, 0, keep);
        rc.v = try rows(g, rc.v, 0, keep);
        rc.len = at - rc.base;
    }

    /// Appends rows from `pos0 == end()`, keeping the `window - 1` rows before them the first new row attends.
    fn append(rc: *RowCache, g: *Graph, k: Array, v: Array, pos0: usize, window: usize) !void {
        if (rc.len == 0) rc.base = pos0;
        if (pos0 != rc.end()) return error.MimoMtpRowGap;
        const n: usize = @intCast(ops.shape(k)[2]);
        const keep = @min(rc.len, window - 1);
        if (keep < rc.len) {
            rc.k = try rows(g, rc.k, @intCast(rc.len - keep), @intCast(rc.len));
            rc.v = try rows(g, rc.v, @intCast(rc.len - keep), @intCast(rc.len));
            rc.base = pos0 - keep;
        }
        if (keep == 0) {
            rc.k = k;
            rc.v = v;
        } else {
            rc.k = try g.concat(&.{ rc.k, k }, 2);
            rc.v = try g.concat(&.{ rc.v, v }, 2);
        }
        rc.len = keep + n;
    }
};

fn rows(g: *Graph, a: Array, start: c_int, stop: c_int) !Array {
    const sh = ops.shape(a);
    return g.slice(a, &.{ 0, 0, start, 0 }, &.{ sh[0], sh[1], stop, sh[3] });
}

/// Appends `ids` to the token ring, dropping from the front past its capacity (a prompt chunk can be wider than
/// the ring); returns the tokens dropped.
fn appendTokens(toks: *[MAX_TOKS]u32, n: *usize, ids: []const u32) usize {
    const total = n.* + ids.len;
    const keep = @min(total, MAX_TOKS);
    const drop = total - keep;
    if (drop >= n.*) {
        @memcpy(toks[0..keep], ids[drop - n.* ..]);
    } else {
        const kept_old = n.* - drop;
        std.mem.copyForwards(u32, toks[0..kept_old], toks[drop..n.*]);
        @memcpy(toks[kept_old..keep], ids);
    }
    n.* = keep;
    return drop;
}

test "mimo mtp: a prompt chunk wider than the token ring keeps its tail" {
    var toks: [MAX_TOKS]u32 = undefined;
    var n: usize = 0;
    var ids: [3 * MAX_TOKS]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i);
    try std.testing.expectEqual(@as(usize, 0), appendTokens(&toks, &n, ids[0..5]));
    try std.testing.expectEqual(@as(usize, 5 + ids.len - MAX_TOKS), appendTokens(&toks, &n, &ids));
    try std.testing.expectEqual(MAX_TOKS, n);
    try std.testing.expectEqualSlices(u32, ids[ids.len - MAX_TOKS ..], toks[0..n]);
    // A narrow append after a full ring keeps the old tail ahead of the new tokens.
    try std.testing.expectEqual(@as(usize, 2), appendTokens(&toks, &n, &.{ 7, 8 }));
    try std.testing.expectEqual(@as(u32, 7), toks[n - 2]);
    try std.testing.expectEqual(ids[ids.len - MAX_TOKS + 2], toks[0]);
}

/// One request's head state, read off the head cache's recurrent entries and written back after every forward:
/// entry k holds head k's keys (`.conv`) and values (`.ssm`); entry 0 also the hidden ring (`.conv_input`) and the
/// integers (`.state_seq`: `meta` then the token ring); entries 1.. hold the round's drafts (`.conv_input`).
const State = struct {
    rows: [MAX_HEADS]RowCache = @splat(.{}),
    /// Rows past this carry a draft token (head k), so a round start drops them.
    committed_end: [MAX_HEADS]usize = @splat(0),
    hid: Array = .{},
    hid_base: usize = 0,
    hid_len: usize = 0,
    toks: [MAX_TOKS]u32 = undefined,
    n_toks: usize = 0,
    tok_base: usize = 0,
    round_q: usize = 0,
    drafts: [MAX_HEADS - 1]Array = @splat(.{}),
    n_drafts: usize = 0,

    const META = 5 + 3 * MAX_HEADS;

    fn load(f: *const abi.Forward, heads: usize) !State {
        var st: State = .{};
        const co = f.cache_ops.?;
        const h = f.caches.slice()[0];
        const meta = co.state(h, 0, .state_seq);
        if (meta.ctx == null) return st;
        const m: Array = .{ .ctx = meta.ctx };
        try mlx.check(mlx.mlx_array_eval(m));
        const n: usize = @intCast(ops.shape(m)[0]);
        const v = (mlx.mlx_array_data_int32(m) orelse return error.MimoMtpStateUnreadable)[0..n];
        st.hid_base = @intCast(v[0]);
        st.hid_len = @intCast(v[1]);
        st.tok_base = @intCast(v[2]);
        st.round_q = @intCast(v[3]);
        st.n_drafts = @intCast(v[4]);
        for (0..heads) |k| {
            st.rows[k] = .{ .base = @intCast(v[5 + 3 * k]), .len = @intCast(v[6 + 3 * k]) };
            st.committed_end[k] = @intCast(v[7 + 3 * k]);
            if (st.rows[k].len > 0) {
                st.rows[k].k = .{ .ctx = co.state(h, @intCast(k), .conv).ctx };
                st.rows[k].v = .{ .ctx = co.state(h, @intCast(k), .ssm).ctx };
            }
        }
        st.n_toks = n - META;
        for (v[META..], 0..) |t, i| st.toks[i] = @intCast(t);
        if (st.hid_len > 0) st.hid = .{ .ctx = co.state(h, 0, .conv_input).ctx };
        for (st.drafts[0..st.n_drafts], 1..) |*d, k| d.* = .{ .ctx = co.state(h, @intCast(k), .conv_input).ctx };
        return st;
    }

    fn save(st: *const State, g: *Graph, f: *const abi.Forward, heads: usize) !void {
        const co = f.cache_ops.?;
        const h = f.caches.slice()[0];
        var v: [META + MAX_TOKS]i32 = undefined;
        v[0] = @intCast(st.hid_base);
        v[1] = @intCast(st.hid_len);
        v[2] = @intCast(st.tok_base);
        v[3] = @intCast(st.round_q);
        v[4] = @intCast(st.n_drafts);
        for (0..MAX_HEADS) |k| {
            v[5 + 3 * k] = @intCast(st.rows[k].base);
            v[6 + 3 * k] = @intCast(st.rows[k].len);
            v[7 + 3 * k] = @intCast(st.committed_end[k]);
        }
        for (st.toks[0..st.n_toks], META..) |t, i| v[i] = @intCast(t);
        const none = try g.zeros(&.{0}, .int32);
        for (0..heads) |k| {
            const rc = &st.rows[k];
            try set(co, h, k, .conv, if (rc.len > 0) rc.k else none);
            try set(co, h, k, .ssm, if (rc.len > 0) rc.v else none);
            if (k >= 1 and k <= st.n_drafts) try set(co, h, k, .conv_input, st.drafts[k - 1]);
        }
        if (st.hid_len > 0) try set(co, h, 0, .conv_input, st.hid);
        try set(co, h, 0, .state_seq, try g.fromSlice(i32, v[0 .. META + st.n_toks], &.{@intCast(META + st.n_toks)}));
    }

    fn set(co: *const abi.CacheOps, h: abi.Handle, k: usize, which: abi.StateSlot, a: Array) !void {
        if (!co.set_state(h, @intCast(k), which, .{ .ctx = a.ctx }).isOk()) return error.MimoMtpStateRefused;
    }

    fn step(st: *const State) usize {
        return st.hid_base + st.hid_len;
    }

    fn tokEnd(st: *const State) usize {
        return st.tok_base + st.n_toks;
    }

    /// Keep only what committed length `len` still vouches for (hiddens below it, tokens up to x_len, head k's rows
    /// whose token is one of those) and drop the round's drafts.
    fn truncate(st: *State, g: *Graph, len: usize) !void {
        if (len < st.step()) {
            if (len <= st.hid_base) {
                st.hid = .{};
                st.hid_base = len;
                st.hid_len = 0;
            } else {
                const sh = ops.shape(st.hid);
                st.hid = try g.slice(st.hid, &.{ 0, 0, 0 }, &.{ 1, @intCast(len - st.hid_base), sh[2] });
                st.hid_len = len - st.hid_base;
            }
        }
        if (len + 1 < st.tokEnd()) {
            if (len + 1 <= st.tok_base) {
                st.n_toks = 0;
                st.tok_base = len + 1;
            } else st.n_toks = len + 1 - st.tok_base;
        }
        for (&st.rows, &st.committed_end, 0..) |*r, *ce, k| {
            ce.* = @min(ce.*, len -| k);
            try r.truncate(g, ce.*);
        }
        st.n_drafts = 0;
    }

    /// Committed hiddens for positions `[p0, p0 + n)` and tokens x_{p0+1..p0+n}.
    fn record(st: *State, g: *Graph, p0: usize, hidden: Array, ids: []const u32) !void {
        try st.truncate(g, p0);
        if (st.hid_len == 0) st.hid_base = p0;
        if (p0 != st.step()) return error.MimoMtpRowGap;
        st.hid = if (st.hid_len == 0) hidden else try g.concat(&.{ st.hid, hidden }, 1);
        st.hid_len += ids.len;
        if (st.n_toks == 0) st.tok_base = p0 + 1;
        if (st.tokEnd() != p0 + 1) return error.MimoMtpRowGap;
        st.tok_base += appendTokens(&st.toks, &st.n_toks, ids);
        if (st.hid_len > RING_ROWS) {
            const drop = st.hid_len - RING_ROWS;
            const sh = ops.shape(st.hid);
            var cut = try g.slice(st.hid, &.{ 0, @intCast(drop), 0 }, &.{ 1, sh[1], sh[2] });
            // A slice pins the rows it views: past a prefill chunk, copied and evaluated it lets go.
            if (drop > RING_ROWS) {
                cut = try g.adopt(try materializedOwnedCopy(g.s, cut));
                const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{cut}, 1);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_async_eval(vec));
            }
            st.hid = cut;
            st.hid_base += drop;
            st.hid_len = RING_ROWS;
        }
    }

    fn hiddenRows(st: *const State, g: *Graph, from: usize, to: usize) !Array {
        if (from < st.hid_base or to > st.step()) return error.MimoMtpHiddenOutOfRing;
        const sh = ops.shape(st.hid);
        return g.slice(st.hid, &.{ 0, @intCast(from - st.hid_base), 0 }, &.{ 1, @intCast(to - st.hid_base), sh[2] });
    }

    fn token(st: *const State, p: usize) !u32 {
        if (p < st.tok_base or p >= st.tokEnd()) return error.MimoMtpTokenOutOfRing;
        return st.toks[p - st.tok_base];
    }
};

/// Head `li` over rows `[pos0, pos0 + R)`: `ids` `[1, R]` int32, `hid` `[1, R, H]` the trunk hiddens. Appends the
/// rows' keys and values; the final-normed `[1, R, H]` rows.
fn layerForward(h: *const Heads, g: *Graph, table: *const Linear, li: usize, cache: *RowCache, ids: Array, hid: Array, pos0: usize) !Array {
    const lw = &h.layers[li];
    const r = ops.shape(ids)[1];
    const hidden = ops.shape(hid)[2];
    const emb = try g.reshape(try gatherRows(g, table, try g.reshape(ids, &.{r}), .bfloat16), &.{ 1, r, hidden });
    const en = try g.rmsNorm(emb, lw.enorm, h.eps);
    const hn = try g.rmsNorm(try g.astype(hid, ops.dtype(emb)), lw.hnorm, h.eps);
    const x = try lw.eh_proj.forward(g, try g.concat(&.{ en, hn }, 2));
    const x2 = try g.add(x, try attention(h, g, lw, cache, try g.rmsNorm(x, lw.input_norm, h.eps), r, pos0));
    const m_in = try g.rmsNorm(x2, lw.pre_mlp_norm, h.eps);
    const gt = try lw.gate.forward(g, m_in);
    const u = try lw.up.forward(g, m_in);
    const mlp = try lw.down.forward(g, try g.mul(try g.mul(gt, try g.sigmoid(gt)), u));
    return g.rmsNorm(try g.add(x2, mlp), lw.final_norm, h.eps);
}

fn attention(h: *const Heads, g: *Graph, lw: *const Layer, cache: *RowCache, x: Array, r: c_int, pos0: usize) !Array {
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const offset: c_int = @intCast(pos0);
    const q = try g.rope(try g.transpose(try g.reshape(try lw.q.forward(g, x), &.{ 1, r, h.n_heads, h.head_dim }), &perm), h.rope_dims, h.rope_base, offset);
    const k = try g.rope(try g.transpose(try g.reshape(try lw.k.forward(g, x), &.{ 1, r, h.n_kv, h.head_dim }), &perm), h.rope_dims, h.rope_base, offset);
    const v = try g.transpose(try g.reshape(try lw.v.forward(g, x), &.{ 1, r, h.n_kv, h.v_dim }), &perm);
    try cache.append(g, k, try g.mul(v, try g.scalar(h.value_scale, ops.dtype(v))), pos0, h.window);
    // Query i (position pos0 + i) sees keys in (its position - window, its position].
    const keys: c_int = @intCast(cache.len);
    const masked = r > 1 or keys > @as(c_int, @intCast(h.window));
    var mask: Array = .{};
    if (masked) {
        const host = try gpa.alloc(bool, @intCast(r * keys));
        defer gpa.free(host);
        const first: i64 = @intCast(cache.base);
        const w: i64 = @intCast(h.window);
        for (0..@intCast(r)) |i| {
            const qp: i64 = @intCast(pos0 + i);
            for (0..@intCast(keys)) |j| {
                const kp = first + @as(i64, @intCast(j));
                host[i * @as(usize, @intCast(keys)) + j] = kp <= qp and kp > qp - w;
            }
        }
        mask = try g.fromSlice(bool, host, &.{ 1, 1, r, keys });
    }
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h.head_dim)));
    const out = try g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, cache.k, cache.v, scale, if (masked) "array" else "", mask, lw.sinks, false });
    return lw.o.forward(g, try g.reshape(try g.transpose(out, &perm), &.{ 1, r, h.n_heads * h.v_dim }));
}

/// Head `li`'s rows from wherever its rows end through `q`, the last `drafted` carrying this round's drafts.
fn forwardRows(h: *const Heads, g: *Graph, table: *const Linear, st: *State, li: usize, q: usize, drafted: usize) !Array {
    const cache = &st.rows[li];
    // Rows before the last query's window never reach a later query either.
    const floor = (q + 1) -| h.window;
    const start = @max(@max(cache.end(), floor), st.hid_base);
    if (cache.end() < start) cache.reset(start);
    if (start > q) return error.MimoMtpNothingToForward;
    const n = q + 1 - start;
    const tentative = @min(drafted, n);
    var host: [RING_ROWS + MAX_HEADS]i32 = undefined;
    if (n - tentative > host.len) return error.MimoMtpChunkTooWide;
    for (host[0 .. n - tentative], 0..) |*t, i| t.* = @intCast(try st.token(start + i + li + 1));
    var parts: [MAX_HEADS + 1]Array = undefined;
    var np: usize = 0;
    if (n > tentative) {
        parts[np] = try g.fromSlice(i32, host[0 .. n - tentative], &.{@intCast(n - tentative)});
        np += 1;
    }
    // Row p's token x_{p+li+1} is draft d_{p+li-q}.
    for (st.drafts[drafted - tentative .. drafted]) |d| {
        parts[np] = d;
        np += 1;
    }
    const ids = try g.reshape(try g.astype(try g.concat(parts[0..np], 0), .int32), &.{ 1, @intCast(n) });
    const out = try layerForward(h, g, table, li, cache, ids, try st.hiddenRows(g, start, q + 1), start);
    st.committed_end[li] = q + 1 - tentative;
    return out;
}

fn lastRow(g: *Graph, x: Array) !Array {
    const sh = ops.shape(x);
    return g.slice(x, &.{ 0, sh[1] - 1, 0 }, &.{ 1, sh[1], sh[2] });
}

/// `HeadHooks.forward`. Without `want_hidden`: committed history (every head appends the rows whose tokens are
/// committed). At head 0: a round opens on the committed rows and runs head 0 at their last row q. Past it: draft
/// step `head` runs at q on the drafts so far plus `ids` (the newest draft; a step past the last head re-runs it).
pub fn forward(h: *const Heads, table: *const Linear, s: ops.Stream, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    var g = Graph.init(gpa, s);
    defer g.deinit();
    if (f.cache_ops == null or f.caches.len != 1) return error.MimoMtpNeedsHeadCache;
    var st = try State.load(f, h.n);
    const hidden: Array = .{ .ctx = f.hidden_in.ctx };
    const p0: usize = f.positions.slice()[0];
    const opens = !f.want_hidden or f.head == 0;
    const ids_arr: Array = if (f.ids.ctx != null) .{ .ctx = f.ids.ctx } else try g.fromSlice(u32, f.tokens.slice(), &.{@intCast(f.rows)});
    var y: Array = .{};
    if (opens) {
        const n: usize = @intCast(f.rows);
        const host = try gpa.alloc(u32, n);
        defer gpa.free(host);
        const i32s = try g.astype(try g.reshape(ids_arr, &.{@intCast(n)}), .int32);
        try mlx.check(mlx.mlx_array_eval(i32s));
        for (host, (mlx.mlx_array_data_int32(i32s) orelse return error.MimoMtpIdsUnreadable)[0..n]) |*o, t| o.* = @intCast(t);
        try st.record(&g, p0, hidden, host);
        // The engine's history length: KV layer 0 counts the committed rows.
        const co = f.cache_ops.?;
        const counted = co.len(f.caches.slice()[0], 0);
        const grew: c_int = @intCast(@as(u64, st.step()) -| counted);
        if (grew > 0) {
            var view: abi.KvView = .{};
            const z = try g.zeros(&.{ 1, 1, grew, 1 }, .bfloat16);
            const v = co.append(f.caches.slice()[0], 0, .{ .ctx = z.ctx }, .{ .ctx = z.ctx }, 1, .{ .ctx = s.ctx }, &view);
            co.release(&view);
            if (!v.isOk()) return error.MimoMtpCountRefused;
        }
        if (!f.want_hidden) {
            const end = st.step();
            for (0..h.n) |li| {
                // Head li's row p needs x_{p+li+1}, committed through x_end.
                if (end <= li or end - li <= st.rows[li].end()) continue;
                _ = try forwardRows(h, &g, table, &st, li, end - li - 1, 0);
            }
        } else {
            st.round_q = st.step() - 1;
            y = try lastRow(&g, try forwardRows(h, &g, table, &st, 0, st.round_q, 0));
        }
    } else {
        const li: usize = @min(f.head, h.n - 1);
        if (li == st.n_drafts + 1) {
            st.drafts[st.n_drafts] = try g.reshape(ids_arr, &.{1});
            st.n_drafts += 1;
        } else if (li != h.n - 1 or st.n_drafts != h.n - 1) return error.MimoMtpStepOutOfOrder;
        try st.rows[li].truncate(&g, st.committed_end[li]);
        y = try lastRow(&g, try forwardRows(h, &g, table, &st, li, st.round_q, @min(st.n_drafts, li)));
    }
    try st.save(&g, f, h.n);
    if (y.ctx != null) out.hidden = .{ .ctx = g.escape(y).ctx };
}
