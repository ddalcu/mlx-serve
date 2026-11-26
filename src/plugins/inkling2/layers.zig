//! Inkling 2's layer, as mlx-lm's `models/inkling2/layers.py`: depth attention over the blocks' saved streams in
//! place of a plain residual, separate query/key/value inputs on a global layer, L2-clipped q/k and relative states,
//! a hidden-conditioned relative bias, a per-row attention sink, a sigmoid output gate, short convs on k, v and both
//! sublayer outputs, the loop's message gates, and a sigmoid-routed MoE under a soft-capped SwiGLU.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const abi = @import("../../abi/plugin.zig");
const band = @import("../../sdk/bases/blocks/sliding.zig");
const head_block = @import("../../sdk/bases/blocks/head.zig");
const sdpaSinks = @import("../../sdk/bases/blocks/decoder_attention.zig").sdpaSinks;
const linear = @import("../../sdk/bases/blocks/linear.zig");
pub const Linear = linear.Linear;
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const device = @import("../../mlx/device.zig");
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const rel = @import("../inkling/attention.zig");
const Config = @import("config.zig").Config;

const Graph = ops.Graph;
const Array = ops.Array;
const Ctx = block.Ctx;

/// The unweighted norms' epsilon (mlx-lm's `unweighted_rms_norm` default), whatever the config's.
const plain_eps: f32 = 1e-6;
/// mlx-lm sorts a call's (token, expert) pairs by expert from this many on.
const sort_from: c_int = 64;

/// A short conv on k or v: one depthwise kernel `[C, K, 1]`, or on a global layer a per-head kernel `[K, C]` plus
/// one projected from the layer's input (`[heads * K, hidden]`).
pub const Conv = union(enum) {
    static: Array,
    dynamic: struct { static: Array, proj: Linear },
};

/// A looped layer's message gates, stacked q, k, v, mlp: `down` `[4, rank, 2 hidden]`, `forget` and `message`
/// `[4, hidden, rank]`, each in its stored dtype (the release's bf16 is read as half the bytes of an f32 copy).
pub const Gates = struct { down: Array, forget: Array, message: Array };

pub const Layer = struct {
    global: bool,
    /// Opens a depth block: its input stream is saved and its attention output starts the new one.
    boundary: bool,
    /// A depth query is held `[hidden, Q]` f32, times hidden^-0.5. This one has 1 or 3 columns; null on layer 0,
    /// which attends its own input.
    attn_query: Array = .{},
    mlp_query: Array,
    /// The q, k and v inputs' norms (one weight thrice on a sliding layer).
    in_norm: [3]Array,
    /// Every projection keeps the checkpoint's layout, `[out, in]` when dense (`projectRows`): MLX's one-row
    /// matmul over a pre-transposed `[in, out]` weight walks its columns, four times the latency.
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    o: Linear,
    gate: Linear,
    sink: Linear,
    /// `[heads, head_dim]` and `[heads, d_rel]`.
    q_scale: Array,
    rel_scale: Array,
    /// `[d_rel, extent]`, its contiguous `[extent, d_rel]` transpose for on-the-fly NAX bias, and the f32 column-reversed
    /// copy (the one-row step's bias, newest key last).
    rel_proj: Array,
    rel_proj_t: Array = .{},
    rel_proj_rev: Array,
    k_conv: Conv,
    v_conv: Conv,
    attn_conv: Array,
    mlp_conv: Array,
    post_norm: Array,
    /// `[experts, hidden]` and `[experts]`, f32.
    router: Array,
    router_bias: Array,
    /// q | gate | r | sink, and a sliding layer's k | v, as one projection each (the parts are its row views).
    join_q: linear.Joined = .{},
    join_kv: linear.Joined = .{},
    /// Rows alternate gate, up.
    gate_up: Bank,
    down: Bank,
    /// NVFP4's per-expert global scale over 6 x 448, `[experts]` f32; null = none.
    gate_up_scale: Array = .{},
    down_scale: Array = .{},
    /// Set on a looped layer, with its q, k, v and mlp message queries `[hidden, 4]`, held as a depth query.
    gates: ?Gates = null,
    message_query: Array = .{},
};

pub const Body = struct {
    cfg: Config,
    layers: []const Layer,
    final_query: Array,
    /// Ones `[hidden]` and `[head_dim]` f32: the unweighted norms' weight.
    ones: Array,
    ones_head: Array,
};

pub const projectRows = head_block.project;

fn plainNorm(g: *Graph, x: Array, ones: Array) !Array {
    return g.rmsNorm(x, ones, plain_eps);
}

/// Attention of the learned `query` (`[D, Q]`, scaled) over the open depth blocks (`stack` `[1, S, N, D]`, null = none) and the
/// live `partial` stream `[1, S, D]`, all f32, the keys and the values: `[1, S, Q, D]`.
pub fn depthAttention(g: *Graph, stack: ?Array, partial: Array, query: Array) !Array {
    const live = try g.expandDims(partial, 2);
    const all = if (stack) |b| try g.concat(&.{ b, live }, 2) else live;
    const weights = try g.softmax(try g.matmul(all, query), 2, true);
    return g.matmul(try g.transpose(weights, &.{ 0, 1, 3, 2 }), all);
}

fn row(g: *Graph, x: Array, i: c_int) !Array {
    const sh = ops.shape(x);
    return g.reshape(try g.sliceAxis(x, 2, i, i + 1), &.{ sh[0], sh[1], sh[3] });
}

/// `(1 - tanh(forget)) * hidden + tanh(gain) * message`, the gates read off both streams' norms, in f32.
fn messageGate(g: *Graph, w: *const Body, gates: *const Gates, i: c_int, hidden: Array, message: Array) !Array {
    const h = try g.astype(hidden, .float32);
    const both = try g.concat(&.{ try plainNorm(g, h, w.ones), try plainNorm(g, message, w.ones) }, 2);
    const st = try g.astype(try g.matmul(try g.astype(both, ops.dtype(gates.down)), try downs(g, gates, i, 1)), .float32);
    const forget = try g.tanh(try g.linear(st, try g.astype(try stacked(g, gates.forget, i), .float32), null));
    const gain = try g.tanh(try g.linear(st, try g.astype(try stacked(g, gates.message, i), .float32), null));
    return g.add(try g.mul(try g.sub(try g.scalar(1, .float32), forget), h), try g.mul(gain, message));
}

/// Gates `from .. from + n`'s down projections as a matmul's right side, `[n, 2 hidden, rank]`.
pub fn downs(g: *Graph, gates: *const Gates, from: c_int, n: c_int) !Array {
    return g.transpose(try g.sliceAxis(gates.down, 0, from, from + n), &.{ 0, 2, 1 });
}

/// Entry `i` of a stack `[N, a, b]`.
fn stacked(g: *Graph, x: Array, i: c_int) !Array {
    const sh = ops.shape(x);
    return g.reshape(try g.sliceAxis(x, 0, i, i + 1), &.{ sh[1], sh[2] });
}

/// `x * scale` clipped to an L2 norm of `threshold` over the last axis, in f32, back in `x`'s dtype.
fn l2Clip(g: *Graph, x: Array, scale: ?Array, threshold: f32) !Array {
    var st = try g.astype(x, .float32);
    if (scale) |sc| st = try g.mul(st, try g.astype(sc, .float32));
    const norm = try g.sqrt(try g.sum(try g.square(st), -1, true));
    const ratio = try g.div(try g.scalar(threshold, .float32), try g.addScalar(norm, 1e-6));
    return g.astype(try g.mul(st, try g.minimum(ratio, try g.scalar(1, .float32))), ops.dtype(x));
}

/// `[prev | x]` in f32, `prev` the last K-1 inputs (null = zeros: the sequence's start).
fn convWindow(g: *Graph, x32: Array, prev: ?Array, kernel: c_int) !Array {
    const sh = ops.shape(x32);
    const left = prev orelse try g.zeros(&.{ sh[0], kernel - 1, sh[2] }, .float32);
    return g.concat(&.{ left, x32 }, 1);
}

/// The window's last K-1 rows as a copy, so the state does not hold the chunk.
fn convState(g: *Graph, window: Array, kernel: c_int) !Array {
    const n = ops.shape(window)[1];
    return g.take(window, try g.arange(@floatFromInt(n - (kernel - 1)), @floatFromInt(n), 1, .int32), 1);
}

const Conved = struct { out: Array, state: Array };

/// A causal depthwise conv plus the input, over sqrt(K + 1); the conv runs in the weight's dtype, as the reference.
fn shortConv(g: *Graph, x: Array, w: Array, prev: ?Array) !Conved {
    const kernel = ops.shape(w)[1];
    const x32 = try g.astype(x, .float32);
    const win = try convWindow(g, x32, prev, kernel);
    const y = try g.conv1d(try g.astype(win, ops.dtype(w)), w, 1, 0, 1, ops.shape(x)[2]);
    const sum = try g.add(try g.astype(y, .float32), x32);
    return .{
        .out = try g.astype(try g.mulScalar(sum, 1.0 / @sqrt(@as(f32, @floatFromInt(kernel + 1)))), ops.dtype(x)),
        .state = try convState(g, win, kernel),
    };
}

/// A global layer's conv on k or v: each head's kernel is its static one plus `taps_proj`, the one projected from
/// the layer's input; tap `lag` weighs the input `lag` rows back. Plus the input, over sqrt(2K + 1).
fn dynamicConv(g: *Graph, x: Array, taps_proj: Array, static: Array, prev: ?Array, heads: c_int) !Conved {
    const kernel = ops.shape(static)[0];
    const sh = ops.shape(x);
    const s = sh[1];
    const ch = sh[2];
    const x32 = try g.astype(x, .float32);
    const window = try convWindow(g, x32, prev, kernel);
    const dw = try g.reshape(try g.astype(taps_proj, .float32), &.{ 1, s, heads, 1, kernel });
    const sw = try g.reshape(try g.transpose(try g.astype(static, .float32), &.{ 1, 0 }), &.{ heads, @divExact(ch, heads), kernel });
    var taps: [64]Array = undefined;
    for (0..@intCast(kernel)) |lag| {
        const from = kernel - 1 - @as(c_int, @intCast(lag));
        taps[lag] = try g.expandDims(try g.sliceAxis(window, 1, from, from + s), 3);
    }
    const windows = try g.reshape(try g.concat(taps[0..@intCast(kernel)], 3), &.{ 1, s, heads, @divExact(ch, heads), kernel });
    const y = try g.reshape(try g.sum(try g.mul(windows, try g.add(sw, dw)), -1, false), &.{ 1, s, ch });
    return .{
        .out = try g.astype(try g.mulScalar(try g.add(y, x32), 1.0 / @sqrt(@as(f32, @floatFromInt(2 * kernel + 1)))), ops.dtype(x)),
        .state = try convState(g, window, kernel),
    };
}

fn kvConv(g: *Graph, conv: *const Conv, x: Array, taps_proj: ?Array, prev: ?Array, heads: c_int) !Conved {
    return switch (conv.*) {
        .static => |w| shortConv(g, x, w, prev),
        .dynamic => |*d| dynamicConv(g, x, taps_proj orelse return error.Inkling2NoConvTaps, d.static, prev, heads),
    };
}

fn stateColumns(g: *Graph, st: ?Array, from: c_int, n: c_int) !?Array {
    return if (st) |x| try g.sliceAxis(x, 2, from, from + n) else null;
}

/// What a visit hands its attention read, with what its MLP half needs carried through. Pure: no cache call.
pub const Pre = struct {
    /// `[1, S, H, hd]`, clipped, not yet log-scaled.
    q: Array,
    /// `[1, Hkv, S, hd]`: the rows the cache appends.
    k: Array,
    v: Array,
    /// The clipped relative states `[1, S, H, d_rel]`; at one row their logits `[H, extent]` f32, distance 0 last.
    rel: Array,
    /// The output gate's sigmoid `[1, S, H * hd]` f32.
    gate: Array,
    /// `[1, S, H]`.
    sinks: Array,
    k_state: Array,
    v_state: Array,
    /// The visit's input stream in f32, and its unweighted norm.
    stream: Array,
    normed: Array,
};

/// A visit's projections off its normed inputs: q, gate, r, sink, k, v, and a global layer's projected k and v conv
/// taps, with the input stream in f32 and its unweighted norm.
pub const Projected = struct { proj: [6]Array, dyn: ?[2]Array, stream: Array, normed: Array };

/// The stream to the attention's projections: depth attention over the blocks, the loop's message gates, the input
/// norms and the projections. Pure: no cache call.
pub fn project(g: *Graph, L: *const Layer, w: *const Body, h: Array, blocks: ?Array, messages: ?Array) !Projected {
    const cfg = &w.cfg;
    const dt = ops.dtype(h);
    const stream = try g.astype(h, .float32);
    const normed = try plainNorm(g, stream, w.ones);

    var qkv = [3]Array{ h, h, h };
    if (L.attn_query.ctx != null) {
        const ai = try depthAttention(g, blocks, normed, L.attn_query);
        const three = ops.shape(ai)[2] == 3;
        for (&qkv, 0..) |*x, i| x.* = try row(g, ai, if (three) @intCast(i) else 0);
    }
    if (messages) |m| {
        const gates = &(L.gates orelse return error.Inkling2NoMessageGate);
        for (&qkv, 0..) |*x, i| x.* = try messageGate(g, w, gates, @intCast(i), x.*, try row(g, m, @intCast(i)));
    }
    if (!L.global and messages == null) {
        // A sliding layer's three inputs are one array under one norm.
        qkv = @splat(try g.astype(try g.rmsNorm(qkv[0], L.in_norm[0], cfg.rms_norm_eps), dt));
    } else for (&qkv, L.in_norm) |*x, nw| {
        x.* = try g.astype(try g.rmsNorm(x.*, nw, cfg.rms_norm_eps), dt);
    }

    const ps = try projections(g, L, qkv);
    return .{ .proj = ps.proj, .dyn = ps.dyn, .stream = stream, .normed = normed };
}

pub const Projections = struct { proj: [6]Array, dyn: ?[2]Array };

/// q, gate, r, sink, k, v off the normed q, k and v inputs, joined where the pack's layout allows it and the inputs
/// are one array; then a global layer's projected k and v conv taps.
pub fn projections(g: *Graph, L: *const Layer, qkv: [3]Array) !Projections {
    var proj: [6]Array = undefined;
    if (L.join_q.built()) {
        const y = try projectRows(g, &L.join_q.lin, qkv[0]);
        for (proj[0..4], 0..) |*x, i| x.* = try L.join_q.part(g, y, i);
    } else for (proj[0..4], [_]*const Linear{ &L.q, &L.gate, &L.r, &L.sink }) |*x, l| {
        x.* = try projectRows(g, l, qkv[0]);
    }
    if (L.join_kv.built() and qkv[1].ctx == qkv[0].ctx and qkv[2].ctx == qkv[0].ctx) {
        const y = try projectRows(g, &L.join_kv.lin, qkv[0]);
        for (proj[4..6], 0..) |*x, i| x.* = try L.join_kv.part(g, y, i);
    } else {
        proj[4] = try projectRows(g, &L.k, qkv[1]);
        proj[5] = try projectRows(g, &L.v, qkv[2]);
    }
    const dyn: ?[2]Array = if (L.k_conv == .dynamic) .{ try projectRows(g, &L.k_conv.dynamic.proj, qkv[1]), try projectRows(g, &L.v_conv.dynamic.proj, qkv[2]) } else null;
    return .{ .proj = proj, .dyn = dyn };
}

/// The projections to what the attention reads: the k and v convs behind the visit's conv state `st` (null = the
/// sequence's start), the clips, the value norm and the gate's sigmoid. Pure: no cache call.
pub fn prepare(g: *Graph, L: *const Layer, w: *const Body, p: *const Projected, st: ?Array) !Pre {
    const cfg = &w.cfg;
    const s = ops.shape(p.stream)[1];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const kv_dim = kv_heads * hd;
    const dt = ops.dtype(p.proj[0]);
    const ks = try kvConv(g, &L.k_conv, p.proj[4], if (p.dyn) |d| d[0] else null, try stateColumns(g, st, 0, kv_dim), kv_heads);
    const vs = try kvConv(g, &L.v_conv, p.proj[5], if (p.dyn) |d| d[1] else null, try stateColumns(g, st, kv_dim, kv_dim), kv_heads);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const k = try l2Clip(g, try g.reshape(ks.out, &.{ 1, s, kv_heads, hd }), null, cfg.qk_clip);
    const v = try g.astype(try g.rmsNorm(try g.astype(try g.reshape(vs.out, &.{ 1, s, kv_heads, hd }), .float32), w.ones_head, cfg.rms_norm_eps), dt);
    const rs = try l2Clip(g, try g.reshape(p.proj[2], &.{ 1, s, heads, @intCast(cfg.d_rel) }), L.rel_scale, cfg.rel_clip);
    return .{
        .q = try l2Clip(g, try g.reshape(p.proj[0], &.{ 1, s, heads, hd }), L.q_scale, cfg.qk_clip),
        .k = try g.transpose(k, &perm),
        .v = try g.transpose(v, &perm),
        .rel = if (s == 1) try g.reshape(try g.matmul(try g.astype(rs, .float32), L.rel_proj_rev), &.{ heads, ops.shape(L.rel_proj)[1] }) else rs,
        .gate = try g.sigmoid(try g.astype(p.proj[1], .float32)),
        .sinks = p.proj[3],
        .k_state = ks.state,
        .v_state = vs.state,
        .stream = p.stream,
        .normed = p.normed,
    };
}

/// The log-scaling factor of the row that makes the sequence `total` long, on a global layer.
pub fn stepTau(cfg: *const Config, total: c_int) f32 {
    if (total <= @as(c_int, @intCast(cfg.log_n_floor))) return 1.0;
    return 1.0 + cfg.log_alpha * @log(@as(f32, @floatFromInt(total)) / @as(f32, @floatFromInt(cfg.log_n_floor)));
}

/// Appends a visit's rows `k`, `v` to `cache`: a sliding visit through its ring (`Config.ring` rows, the older ones
/// dropped), a global one keeping every row. The view is the newest `keep` rows (0 = every row held).
pub fn appendKv(c: *const Ctx, cache: abi.Handle, global: bool, cfg: *const Config, k: Array, v: Array, keep: u32) !abi.KvView {
    if (global or cfg.ring == 0) return kv.appendSliding(c, cache, k, v, keep, 0, 0);
    return kv.appendSliding(c, cache, k, v, keep, cfg.sliding_window, cfg.ring);
}

/// The attention read: appends the visit's k and v, then attends under the relative bias, the sink and the output
/// gate. Returns the gated rows `[1, S, H * hd]` in the stream's dtype.
pub fn attend(c: *Ctx, L: *const Layer, w: *const Body, p: *const Pre) !Array {
    const g = c.g;
    const cfg = &w.cfg;
    const s = c.rows;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const offset: c_int = @intCast(c.offset());
    const total = offset + s;
    const dt = ops.dtype(p.k);
    const span: u32 = if (L.global) 0 else cfg.sliding_window;
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q = try g.transpose(p.q, &perm);

    const sv = band.view(span, hd, total, s);
    var view = try appendKv(c, c.cache(), L.global, cfg, p.k, p.v, if (span > 0) sv.span else 0);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    const kv_len = ops.shape(dv.k)[2];
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // The rows' reads `[1, S, H, hd]` (one row's `[1, H, 1, hd]` holds the same order).
    var attn: Array = undefined;
    if (s == 1) {
        // One row sees every key of its view, the newest last: its bias is the logits' tail, zero past the extent.
        const extent = ops.shape(p.rel)[1];
        var bias = if (kv_len < extent) try g.sliceAxis(p.rel, 1, extent - kv_len, extent) else p.rel;
        if (kv_len > extent) bias = try g.padAxis(bias, 1, kv_len - extent, 0);
        if (L.global and total > @as(c_int, @intCast(cfg.log_n_floor))) {
            const tau = stepTau(cfg, total);
            q = try g.astype(try g.mulScalar(try g.astype(q, .float32), tau), dt);
            bias = try g.mulScalar(bias, tau);
        }
        const mask = try g.astype(try g.reshape(bias, &.{ 1, heads, 1, kv_len }), dt);
        attn = try sdpaSinks(g, q, dv.k, dv.v, scale, "array", mask, try g.reshape(p.sinks, &.{heads}), false);
    } else attn = try wideRead(g, L, cfg, p, dv.k, dv.v, offset);
    const flat = try g.astype(try g.reshape(attn, &.{ 1, s, heads * hd }), .float32);
    return g.astype(try g.mul(flat, p.gate), dt);
}

/// A prompt's rows `[1, S, H, hd]` reading `k` and `v` (the view after the append, the rows at its last S keys) as
/// an op chain: the relative bias folded into one additive mask with the causal band, a zero key and value whose
/// mask entry is each row's sink. `offset` is the first row's position.
pub fn wideRead(g: *Graph, L: *const Layer, cfg: *const Config, p: *const Pre, k: Array, v: Array, offset: c_int) !Array {
    const s = ops.shape(p.q)[1];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const dt = ops.dtype(k);
    const kv_len = ops.shape(k)[2];
    const kv_start = offset + s - kv_len;
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q = try g.transpose(p.q, &perm);
    // The relative bias where a key is visible, a large negative elsewhere.
    const bias = try g.adopt(try rel.inklingRelBias(g.s, p.rel, L.rel_proj, offset, kv_len, kv_start));
    var mask = try g.adopt(try rel.inklingCombineMask(g.s, bias, offset, kv_len, kv_start, if (L.global) null else @intCast(cfg.sliding_window)));
    if (L.global and offset + s > @as(c_int, @intCast(cfg.log_n_floor))) {
        const tau = try g.adopt(try rel.inklingLogScaleTau(g.s, offset, s, cfg.log_n_floor, cfg.log_alpha));
        q = try g.astype(try g.mul(try g.astype(q, .float32), tau), dt);
        mask = try g.mul(mask, tau);
    }
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const sink_col = try g.reshape(try g.transpose(try g.astype(p.sinks, .float32), &.{ 0, 2, 1 }), &.{ 1, heads, s, 1 });
    const zero = try g.zeros(&.{ 1, kv_heads, 1, hd }, dt);
    const m = try g.astype(try g.concat(&.{ mask, sink_col }, 3), dt);
    const attn = try g.sdpa(q, try g.concat(&.{ k, zero }, 2), try g.concat(&.{ v, try g.astype(zero, ops.dtype(v)) }, 2), 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .{ .array = m });
    return g.transpose(attn, &perm);
}

/// `x` through the experts `sel` picks, as the bank's matmul leaves it: `x` `[N, 1, in]` (one row per pick) or
/// `[1, 1, in]` (one row over every pick), `[N, 1, out]`.
pub fn gather(g: *Graph, x: Array, bank: *const Bank, sel: Array, sorted: bool) !Array {
    const dt = ops.dtype(x);
    const rows = gatherRowsDtype(dt, bank.qp.mode, mlx.streamIsGpu(g.s), device.deviceGeneration().gen);
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, if (rows == dt) x else try g.astype(x, rows), bank.w, bank.s, bank.b, .{ .ctx = null }, sel, bank.qp.bits, bank.qp.group_size, bank.qp.mode, sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    const out = try g.adopt(y);
    return if (rows == dt) out else g.astype(out, dt);
}

/// The dtype a gather reads its rows in. A GPU before Apple9 (M1 g13, M2 g14) has no bf16 ALU, and MLX's gather over
/// bf16 rows runs at about half the speed of fp16 ones: an affine bank's bf16 rows go in as fp16 (exact over the
/// stream's range) and the result rounds back. The release's NVFP4 and MXFP8 banks stay bf16: their codes times the
/// block scales carry the per-expert multiplier's inverse, so raw outputs overflow fp16 (the answer was `!!`). An
/// unknown generation (0) and a CPU stream keep the rows' own dtype.
pub fn gatherRowsDtype(dt: ops.Dtype, mode: QuantMode, gpu: bool, gen: u32) ops.Dtype {
    return if (dt == .bfloat16 and mode == .affine and gpu and gen != 0 and gen < 15) .float16 else dt;
}

/// `gather` in f32, scaled per expert.
fn expertMm(g: *Graph, x: Array, bank: *const Bank, scale: Array, sel: Array, sorted: bool) !Array {
    const out = try g.astype(try gather(g, x, bank, sel, sorted), .float32);
    return if (scale.ctx == null) out else g.mul(out, try g.reshape(try g.take(scale, sel, 0), &.{ ops.shape(sel)[0], 1, 1 }));
}

/// The router's picks `[1, S, top_k]` u32 and their weights (f32): the top-k of sigmoid scores plus the selection
/// bias, weighted by the unbiased scores over their sum times `route_scale`.
pub const Route = struct { idx: Array, weights: Array };

pub fn route(g: *Graph, L: *const Layer, cfg: *const Config, x: Array) !Route {
    const top_k: c_int = @intCast(cfg.top_k);
    const scores = try g.sigmoid(try g.matmul(try g.astype(x, .float32), try g.transpose(L.router, &.{ 1, 0 })));
    const picked = try g.argpartition(try g.neg(try g.add(scores, L.router_bias)), top_k - 1, -1);
    const idx = try g.astype(try g.sliceAxis(picked, 2, 0, top_k), .uint32);
    const raw = try g.takeAlong(scores, idx, -1);
    return .{ .idx = idx, .weights = try g.mulScalar(try g.div(raw, try g.addScalar(try g.sum(raw, -1, true), 1e-20)), cfg.route_scale) };
}

/// A call's (row, pick) pairs as the banks read them: `rows` `[N, 1, D]` (or one row broadcast over its picks) and
/// their experts `sel` `[N]`. From `sort_from` pairs on they are sorted by expert, so a wide call reads one bank at
/// a time (mlx-lm's `_gather_sort`), with `inverse` the place of pair (row, pick) in that order.
pub const Pairs = struct { rows: Array, sel: Array, inverse: ?Array };

pub fn pairs(g: *Graph, cfg: *const Config, x: Array, idx: Array) !Pairs {
    const sh = ops.shape(x);
    const s = sh[1];
    const d = sh[2];
    const top_k: c_int = @intCast(cfg.top_k);
    const n = s * top_k;
    if (n < sort_from) return .{
        .sel = try g.reshape(idx, &.{n}),
        .rows = if (s == 1) x else try g.reshape(try g.repeat(try g.reshape(x, &.{ s, 1, d }), top_k, 1), &.{ n, 1, d }),
        .inverse = null,
    };
    const flat = try g.reshape(idx, &.{n});
    const order = try g.argsort(flat, 0);
    const token = try g.call(mlx.mlx_floor_divide, .{ order, try g.astype(try g.scalar(@floatFromInt(top_k), .float32), ops.dtype(order)) });
    return .{
        .sel = try g.take(flat, order, 0),
        .rows = try g.reshape(try g.take(try g.reshape(x, &.{ s, d }), token, 0), &.{ n, 1, d }),
        .inverse = try g.argsort(order, 0),
    };
}

/// The picked experts over `x`: gate_up, the soft-capped SwiGLU `cap_g tanh(g / cap_g) sigmoid(g) cap_u tanh(u /
/// cap_u)` and down per pick, weighted and summed, in `x`'s dtype.
fn expertSum(g: *Graph, L: *const Layer, cfg: *const Config, x: Array, r: Route) !Array {
    const p = try pairs(g, cfg, x, r.idx);
    const sh = ops.shape(x);
    const s = sh[1];
    const d = sh[2];
    const top_k: c_int = @intCast(cfg.top_k);
    const inter: c_int = @intCast(cfg.moe_intermediate_size);
    const n = s * top_k;
    const sorted = p.inverse != null;
    const gu = try g.reshape(try expertMm(g, p.rows, &L.gate_up, L.gate_up_scale, p.sel, sorted), &.{ n, inter, 2 });
    const gate = try g.reshape(try g.sliceAxis(gu, 2, 0, 1), &.{ n, 1, inter });
    const up = try g.reshape(try g.sliceAxis(gu, 2, 1, 2), &.{ n, 1, inter });
    const capped = try g.mul(
        try g.mulScalar(try g.tanh(try g.mulScalar(gate, 1.0 / cfg.gate_softcap)), cfg.gate_softcap),
        try g.mulScalar(try g.tanh(try g.mulScalar(up, 1.0 / cfg.up_softcap)), cfg.up_softcap),
    );
    const act = try g.astype(try g.mul(capped, try g.sigmoid(gate)), ops.dtype(x));
    var out = try g.reshape(try expertMm(g, act, &L.down, L.down_scale, p.sel, sorted), &.{ n, d });
    if (p.inverse) |inverse| out = try g.take(out, inverse, 0);
    const summed = try g.sum(try g.mul(try g.reshape(out, &.{ 1, s, top_k, d }), try g.expandDims(r.weights, -1)), 2, false);
    return g.astype(summed, ops.dtype(x));
}

/// A visit between its attention and its experts: the MLP's input `x` and the router's picks for it, the f32
/// residual the MLP's output joins, the block stack when the layer opened a block, and the attention conv's window.
pub const Mid = struct { x: Array, idx: Array, weights: Array, residual: Array, blocks: ?Array, attn_state: Array };

/// The gated attention rows to the MLP's input: the output projection and its conv, the block boundary or the
/// residual, depth attention, the loop's message gate, the norm and the router. Pure: no cache call.
pub fn mid(g: *Graph, L: *const Layer, w: *const Body, attn: Array, stream: Array, normed: Array, blocks: ?Array, st: ?Array, messages: ?Array) !Mid {
    const cfg = &w.cfg;
    const dt = ops.dtype(attn);
    const kv_dim: c_int = @intCast(cfg.kvDim());
    const hidden: c_int = @intCast(cfg.hidden_size);
    const ac = try shortConv(g, try projectRows(g, &L.o, attn), L.attn_conv, try stateColumns(g, st, 2 * kv_dim, hidden));

    var residual = try g.astype(ac.out, .float32);
    var stack = blocks;
    var opened: ?Array = null;
    if (L.boundary) {
        const saved = try g.expandDims(normed, 2);
        opened = if (blocks) |stack_in| try g.concat(&.{ stack_in, saved }, 2) else saved;
        stack = opened;
    } else residual = try g.add(stream, residual);

    var m = try row(g, try depthAttention(g, stack, try plainNorm(g, residual, w.ones), L.mlp_query), 0);
    if (messages) |msg| m = try messageGate(g, w, &(L.gates orelse return error.Inkling2NoMessageGate), 3, m, try row(g, msg, 3));
    const x = try g.astype(try g.rmsNorm(m, L.post_norm, cfg.rms_norm_eps), dt);
    const r = try route(g, L, cfg, x);
    return .{ .x = x, .idx = r.idx, .weights = r.weights, .residual = residual, .blocks = opened, .attn_state = ac.state };
}

/// A visit's output stream and its MLP conv's window.
pub const Tail = struct { h: Array, mlp_state: Array };

/// The experts, the MLP's conv and the final add. Pure: no cache call.
pub fn tail(g: *Graph, L: *const Layer, w: *const Body, m: *const Mid, st: ?Array) !Tail {
    const cfg = &w.cfg;
    const kv_dim: c_int = @intCast(cfg.kvDim());
    const hidden: c_int = @intCast(cfg.hidden_size);
    const y = try expertSum(g, L, cfg, m.x, .{ .idx = m.idx, .weights = m.weights });
    const mc = try shortConv(g, y, L.mlp_conv, try stateColumns(g, st, 2 * kv_dim + hidden, hidden));
    return .{ .h = try g.astype(try g.add(m.residual, try g.astype(mc.out, .float32)), ops.dtype(m.x)), .mlp_state = mc.state };
}

/// A visit's output stream, and the block stack after it when the layer opened a block.
pub const Post = struct { h: Array, blocks: ?Array };

/// One layer visit. `blocks` `[1, S, N, D]` are the depth blocks open so far (null = none); `messages`
/// `[1, S, 4, D]` on a later loop pass. Returns the new stream and, on a boundary layer, the stack it leaves.
pub fn layerForward(L: *const Layer, w: *const Body, c: *Ctx, h: Array, blocks: ?Array, messages: ?Array) !Post {
    const g = c.g;
    const st = try kv.state(c, .conv);
    const p = try prepare(g, L, w, &try project(g, L, w, h, blocks, messages), st);
    const m = try mid(g, L, w, try attend(c, L, w, &p), p.stream, p.normed, blocks, st, messages);
    const t = try tail(g, L, w, &m, st);
    try kv.setState(c, .conv, try g.concat(&.{ p.k_state, p.v_state, m.attn_state, t.mlp_state }, 2));
    return .{ .h = t.h, .blocks = m.blocks };
}

/// The LM head, soft-capped `cap tanh(logits / cap)` in f32, then the padded vocab's real rows.
pub const Head = struct {
    pub const kind: block.Kind = .head;
    pub const supports_batched_rows = false;
    pub const Weights = struct { lin: Linear, vocab: c_int, softcap: f32 };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        // Trimmed before the cap: the cap's result is then a dense array, not a strided view of the padded rows.
        const logits = try g.astype(try g.sliceAxis(try head_block.project(g, &w.lin, x), 2, 0, w.vocab), .float32);
        return g.mulScalar(try g.tanh(try g.mulScalar(logits, 1.0 / w.softcap)), w.softcap);
    }
};

/// The final stream: depth attention of the output query over every block and the last stream.
pub fn finalInput(g: *Graph, w: *const Body, h: Array, blocks: ?Array) !Array {
    return row(g, try depthAttention(g, blocks, try plainNorm(g, try g.astype(h, .float32), w.ones), w.final_query), 0);
}

/// A loop pass's messages for looped layer `L`: its four queries over `bank` (the blocks the last pass opened after
/// its first) and that pass's normed output `partial`.
pub fn messagesFor(g: *Graph, L: *const Layer, bank: Array, partial: Array) !Array {
    return depthAttention(g, bank, partial, L.message_query);
}

pub fn passOutput(g: *Graph, w: *const Body, h: Array) !Array {
    return plainNorm(g, try g.astype(h, .float32), w.ones);
}
