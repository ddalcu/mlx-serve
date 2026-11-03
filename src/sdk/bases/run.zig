//! What the bases share around their layer loops: one forward's arguments (`Run`), its results (`Out`), the head and
//! the logits rows, layer capture, and the model table's `forward` over any base (`tableForward`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const block = @import("block.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const spliceRows = @import("../vision/splice.zig").spliceRows;
const row_exact = @import("blocks/row_exact.zig");
const prefillHeadDimFused = @import("../blocks/attention/attn256.zig").prefillHeadDimFused;

/// One forward's arguments.
pub const Run = struct {
    mode: abi.Mode,
    /// Tokens per slot.
    rows: c_int,
    slots: c_int = 1,
    /// Each slot's first position.
    positions: []const u32,
    caches: []const abi.Handle = &.{},
    cache_ops: ?*const abi.CacheOps = null,
    logits: abi.Logits = .last_row,
    /// Return the final-normed hidden rows too.
    want_hidden: bool = false,
    /// Layers whose output rows the forward publishes into `captured` (owned by the forward's graph).
    capture_layers: []const u32 = &.{},
    captured: []Array = &.{},
    /// An encoder batch's additive key mask; null = none.
    mask: Array = .{},
    /// `Ctx.mask_add` (`Forward.mask_add`).
    mask_add: Array = .{},
    /// Evaluate the residual after every this many layers and after the last (0 = never; `Forward.eval_every`).
    eval_every: u32 = 0,
    /// `Ctx.capture_spec`.
    capture_spec: bool = false,
    /// `Ctx.row_exact`.
    row_exact: row_exact.Lane = .off,
    /// `Ctx.mrope`: the chunk's M-RoPE cos/sin (`Forward.mrope`).
    mrope: ?[2]Array = null,
    /// `Ctx.tree` (`Forward.tree`).
    tree: ?block.Tree = null,
    /// Hand the residual to the GPU every this many layers (`Forward.pipeline_every`; 0 = never).
    pipeline_every: u32 = 0,
    /// Media rows spliced after the embedding at the placeholder ids, from row `media_offset` (`Forward.media`).
    media: Array = .{},
    media_offset: u32 = 0,
    placeholders: abi.PlaceholderIds = .{},
    /// `Forward.embeds`.
    embeds: Array = .{},
    /// `Ctx.offload` (`Forward.offload`).
    offload: ?*const abi.Offload = null,

    pub fn ctx(r: *const Run, g: *Graph, shared: *const block.Shared) Ctx {
        return .{ .g = g, .mode = r.mode, .rows = r.rows, .slots = r.slots, .positions = r.positions, .caches = r.caches, .cache_ops = r.cache_ops, .shared = shared, .mask = r.mask, .mask_add = r.mask_add, .capture_spec = r.capture_spec, .mrope = r.mrope, .row_exact = r.row_exact, .offload = r.offload, .tree = r.tree };
    }

    /// The embedded rows `h` with the media rows at their placeholders (a prompt chunk only, as the engine splices).
    pub fn splice(r: *const Run, g: *Graph, h: Array, ids: Array) !Array {
        const p = r.placeholders;
        if (r.media.ctx == null or r.rows <= 1 or p.image | p.audio | p.video | p.speech == 0) return h;
        var own = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&own, h));
        return g.adopt(try spliceRows(own, ids, r.media, &.{ p.image, p.audio, p.video, p.speech }, r.media_offset, g.s));
    }

    /// The forward's input rows: `Forward.embeds` when given, else `embedded` (the ids' embedding) with the media.
    pub fn input(r: *const Run, g: *Graph, embedded: Array, ids: Array) !Array {
        if (r.embeds.ctx == null) return r.splice(g, embedded, ids);
        // The caller keeps its rows: the forward drops the graph's own reference as the residual moves on.
        var own = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&own, r.embeds));
        return g.adopt(own);
    }

    /// Publishes `h`, layer `layer`'s output, where it was asked for.
    pub fn capture(r: *const Run, g: *Graph, layer: u32, h: Array) !void {
        for (r.capture_layers, 0..) |id, k| if (id == layer) {
            r.captured[k] = try g.keep(h);
        };
    }

    /// Evaluates `h`, layer `layer` of `n`'s output, at the forward's eval cadence, with every slot's conv state: a
    /// conv state is a copy off the chunk, outside the residual's graph, and left lazy it holds the whole chunk.
    pub fn cadence(r: *const Run, layer: usize, n: usize, h: Array) !void {
        if (r.eval_every == 0 and r.pipeline_every > 0 and (layer + 1) % r.pipeline_every == 0 and layer + 1 < n) {
            const v = mlx.mlx_vector_array_new_value(h);
            defer _ = mlx.mlx_vector_array_free(v);
            return mlx.check(mlx.mlx_async_eval(v));
        }
        if (r.eval_every == 0 or ((layer + 1) % r.eval_every != 0 and layer + 1 != n)) return;
        const vec = mlx.mlx_vector_array_new_value(h);
        defer _ = mlx.mlx_vector_array_free(vec);
        if (r.cache_ops) |co| for (r.caches) |cache| for (0..n) |l| {
            const st = co.state(cache, @intCast(l), .conv);
            if (st.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, .{ .ctx = st.ctx });
        };
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// The head over every row (a row's logits must not depend on how many rows ride with it), then the rows asked for.
    pub fn finish(r: *const Run, g: *Graph, c: *Ctx, final: Array, comptime Head: type, head: *const Head.Weights) !Out {
        var out: Out = .{};
        if (r.want_hidden) out.hidden = final;
        if (r.logits == .none) return out;
        const logits = try Head.forward(head, c, final);
        const sh = ops.shape(logits);
        out.logits = switch (r.logits) {
            .last_row => try g.reshape(try g.sliceAxis(logits, 1, sh[1] - 1, sh[1]), &.{ sh[0], sh[2] }),
            .all_rows => try g.reshape(logits, &.{ sh[0] * sh[1], sh[2] }),
            .none => unreachable,
        };
        return out;
    }
};

/// A forward's results, owned by its graph: logits `[slots, vocab]` (last row) or `[slots * rows, vocab]`, and the
/// final-normed hidden `[slots, rows, hidden]` when asked.
pub const Out = struct {
    logits: Array = .{},
    hidden: Array = .{},
};

/// A loaded model of base `Base`: its weights, the shared values, and where its forwards run.
pub fn Model(comptime Base: type) type {
    return struct {
        weights: Base.Weights,
        shared: block.Shared,
        a: std.mem.Allocator,
        s: ops.Stream,
        /// The placeholder ids media rows splice at (`Parsed.placeholder_ids`).
        placeholders: abi.PlaceholderIds = .{},
        /// The lane `Forward.row_exact` turns on; `off` = the weights are outside the row kernels.
        row_exact: row_exact.Lane = .off,

        pub fn of(m: *anyopaque) *@This() {
            return @ptrCast(@alignCast(m));
        }
    };
}

fn failed(e: anyerror) abi.Verdict {
    return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
}

/// The modes `Base` serves.
pub fn declares(comptime Base: type, mode: abi.Mode) bool {
    return abi.declaresMode(.{ .batches_decode = Base.batches_decode }, mode);
}

/// `ModelTable.forward` over a model of `Base`: checks the mode and the shapes, runs the forward in its own graph and
/// hands every result out as a new reference.
pub fn tableForward(comptime Base: type, m: *Model(Base), f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    if (!declares(Base, f.mode)) return abi.Verdict.refuse(.unsupported, "forward mode not declared");
    // An encoder batch (no cache) carries one position per sequence.
    const slots: usize = if (f.mode == .batched_rows) f.caches.len else if (f.cache_ops == null) @max(1, f.positions.len) else 1;
    const n_ids: usize = if (f.ids.ctx != null) idCount(f.ids) else f.tokens.len;
    if (f.rows == 0 or n_ids != slots * f.rows or f.positions.len < slots) return abi.Verdict.refuse(.failed, "tokens, rows and positions disagree");
    if (f.mode == .decode and f.rows != 1) return abi.Verdict.refuse(.failed, "a decode step is one row");
    if (f.tree.table.ctx != null and f.mode != .verify) return abi.Verdict.refuse(.failed, "a draft tree is a solo verify");
    run(Base, m, f, out, slots) catch |e| return failed(e);
    return .ok;
}

fn idCount(ids: abi.Array) usize {
    var n: usize = 1;
    for (ops.shape(.{ .ctx = ids.ctx })) |d| n *= @intCast(d);
    return n;
}

/// `[slots, rows]` ids owned by `g`: the engine's device array as it is (a lazy sampled token stays lazy), else the
/// host tokens.
fn tokenIds(g: *Graph, a: std.mem.Allocator, f: *const abi.Forward, slots: usize) !Array {
    const sh = [_]c_int{ @intCast(slots), @intCast(f.rows) };
    if (f.ids.ctx != null) return g.reshape(.{ .ctx = f.ids.ctx }, &sh);
    const tokens = f.tokens.slice();
    const buf = try a.alloc(i32, tokens.len);
    defer a.free(buf);
    for (tokens, buf) |t, *i| i.* = @intCast(t);
    return g.fromSlice(i32, buf, &sh);
}

/// Half `i` (0 cos, 1 sin) of the stacked `[2, ...]` M-RoPE angles.
fn mropeHalf(g: *Graph, x: Array, i: c_int) !Array {
    return g.reshape(try g.sliceAxis(x, 0, i, i + 1), ops.shape(x)[1..]);
}

fn run(comptime Base: type, m: *Model(Base), f: *const abi.Forward, out: *abi.ForwardOut, slots: usize) !void {
    var g = Graph.init(m.a, m.s);
    defer g.deinit();
    const ids = try tokenIds(&g, m.a, f, slots);
    const caps = f.capture_layers.slice();
    const captured = try m.a.alloc(Array, caps.len);
    defer m.a.free(captured);
    @memset(captured, .{});
    const r: Run = .{
        .mode = f.mode,
        .rows = @intCast(f.rows),
        .slots = @intCast(slots),
        .positions = f.positions.slice()[0..slots],
        .caches = f.caches.slice(),
        .cache_ops = f.cache_ops,
        .logits = f.logits,
        .want_hidden = f.want_hidden,
        .capture_layers = caps,
        .captured = captured,
        .eval_every = if (f.rows > 1) f.eval_every else 0,
        // Verify rows roll back on a partial accept; a batched decode step (one row) never does.
        .capture_spec = f.mode == .verify or (f.mode == .batched_rows and f.rows > 1),
        .mrope = if (f.mrope.ctx != null) .{ try mropeHalf(&g, .{ .ctx = f.mrope.ctx }, 0), try mropeHalf(&g, .{ .ctx = f.mrope.ctx }, 1) } else null,
        .media = .{ .ctx = f.media.ctx },
        .mask = if (f.key_mask.ctx != null) try g.reshape(.{ .ctx = f.key_mask.ctx }, &.{ @intCast(slots), 1, 1, @intCast(f.rows) }) else .{},
        .mask_add = .{ .ctx = f.mask_add.ctx },
        .media_offset = f.media_offset,
        .placeholders = m.placeholders,
        .embeds = .{ .ctx = f.embeds.ctx },
        .row_exact = if (f.row_exact) m.row_exact else .off,
        .offload = f.offload,
        .tree = if (f.tree.table.ctx == null) null else .{
            .table = .{ .ctx = f.tree.table.ctx },
            .attn = .{ .depth = .{ .ctx = f.tree.depth.ctx }, .path = .{ .ctx = f.tree.path.ctx }, .max_depth = f.tree.max_depth },
        },
        .pipeline_every = f.pipeline_every,
    };
    const res = try Base.forward(&m.weights, &m.shared, &g, ids, r);
    if (caps.len > 0) {
        const dst = out.captured orelse return error.NoCaptureBuffer;
        for (captured, 0..) |x, k| dst[k] = if (x.ctx != null) .{ .ctx = g.escape(x).ctx } else .{};
    }
    out.logits = if (res.logits.ctx != null) .{ .ctx = g.escape(res.logits).ctx } else .{};
    out.hidden = if (res.hidden.ctx != null) .{ .ctx = g.escape(res.hidden).ctx } else .{};
}

/// Forwards narrower than this skip the mid-loop eval cadence entirely.
/// Spec-decode VERIFY forwards (PLD/drafter/MTP: seq 2..~9) ride the
/// prefill layer loops because they're multi-token, but their lazy-graph
/// transients are decode-scale (KB-MB) — the periodic eval() that bounds
/// GB-scale prefill-chunk transients only costs them synchronous
/// pipeline drains. Measured (qwen3.6-27B, 64 layers, cadence 4 = 16
/// drains per MTP round): depth-1 rounds 48 ms where the AR forward is
/// ~34 ms, and the drain tax grows superlinearly with draft depth.
/// seq-1 decode never ran cadence evals, so exempting verify-width
/// forwards bounds nothing that wasn't already unbounded at decode.
pub const PREFILL_EVAL_MIN_SEQ: c_int = 32;

pub const RECURRENCE_EVAL_INTERVAL: usize = 32;

pub const EVAL_EVERY_N_LAYERS: u32 = 48;
/// One value for the prefill loops AND the admission bill: at most `cadence + 1`
/// linear layers hold a chunk-wide stream between cadence points.
pub const PREFILL_EVAL_CADENCE_DEFAULT: u32 = 4;

/// Per-layer prefill transient (bytes) above which the layer loop eval()s
/// after EVERY layer instead of the coarse default cadence. Between eval
/// points ~3 layers' transients coexist under MLX lazy eval, so a
/// multi-GB transient triples; eval-per-layer bounds peak to ~1 layer.
/// Two transients count (both scale with total_kv, so short prompts keep
/// the coarse cadence and pay nothing):
///   - the composed-SDPA score tensor [heads, chunk, total_kv] — only for
///     head_dims no fused kernel covers (prefillHeadDimFused: <= 128 via
///     MLX, 256 via msv_attn_p256; unfused only via the kill switch or an
///     exotic dim);
///   - the dense-fp16 rebuild of a quantized KV cache (denseView runs
///     per layer under --kv-quant, over the FULL cache).
/// Budget calibrated LIVE on gemma-4-26B-A4B-qat-4bit (M4 Max):
/// eval-per-layer costs ~4.5% prefill at a 5K prompt (845 MB scores —
/// below this budget, coarse cadence kept) and ~0% at 102K (3.4 GB
/// scores — flips to per-layer, peak 51.6 -> 27.0 GB, prefill +14%).
/// 2 GiB puts the flip at ~8K tokens on 16-head/256-hd geometry, where
/// the ~3% sync cost buys a 13.7 -> 4.3 GB transient bound — the margin
/// that keeps a 32 GB Mac serving a 26B alive.
pub const PREFILL_EVAL_TRANSIENT_BUDGET: u64 = 2 << 30;

/// Pure cadence pick for the prefill layer loops (standard/MoE/hybrid).
pub fn prefillEvalCadence(
    default_cadence: u32,
    score_head_dim: u32,
    head_dim: u32,
    n_heads: u32,
    kv_heads: u32,
    chunk_len: u64,
    total_kv: u64,
    kv_dequant: bool,
) u32 {
    const scores: u64 = if (score_head_dim > 0 and !prefillHeadDimFused(score_head_dim)) @as(u64, n_heads) * chunk_len * total_kv * 2 else 0;
    const dequant: u64 = if (kv_dequant) 2 * total_kv * @as(u64, kv_heads) * @as(u64, head_dim) * 2 else 0;
    return if (scores + dequant > PREFILL_EVAL_TRANSIENT_BUDGET) 1 else default_cadence;
}

/// A trailing remainder smaller than this merges into the preceding chunk
/// instead of becoming its own chunk. Chat-templated prompts routinely land a
/// token or two past a chunk multiple (an "8192-token" prompt tokenizes to
/// 8193); a 1-token final chunk pays a full graph build + eval barrier +
/// cache clear for one token. The merged chunk's attention-score transient
/// grows by at most TAIL_MERGE_MAX/default_chunk (~6% at 8192) — within the
/// score-budget slack `boundedPrefillChunk` already carries.
pub const TAIL_MERGE_MAX: usize = 512;

/// Slots one batched forward carries (twin of `scheduler.MAX_BATCH_GROUP`).
pub const MAX_BATCH_ROWS = 32;
