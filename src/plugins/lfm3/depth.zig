//! LFM3's depth decoder, mlx-vlm's `lfm3_audio/depth.py`: one speech frame (one code per codebook) from the trunk's
//! last hidden row. `depth_linear` lays the row over one slot per codebook; each MaskGIT pass adds every slot's revealed
//! code embedding (else the mask embedding), runs a bidirectional LFM3 trunk (`trunk.zig`) and reads each slot's tied
//! head behind an RMS norm scaled in f32. Pass p reveals `schedule[p]` codebooks of the frame's order. A frame's input
//! to the trunk is the sum of its codes' rows in the model's `codebook_embedding`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const trunk_mod = @import("trunk.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

pub const max_codebooks = 32;

/// How one frame samples: the speech sampling of the request (`audio_temperature`, `audio_top_k`,
/// `audio_repetition_penalty`), and the frame's MaskGIT reveal order and random key.
pub const Sampling = struct {
    temperature: f32,
    /// 0 = every code.
    top_k: u32 = 0,
    repetition_penalty: f32 = 1.0,
    /// A permutation of the codebooks; the first `schedule[0]` are revealed by the first pass, and so on.
    order: []const u32,
    key: u64,
};

pub const Depth = struct {
    linear: Linear,
    mask: Array,
    trunk: trunk_mod.Trunk,
    /// The codebooks' embedding tables stacked `[K * V, D]` (the slots' inputs) and as heads `[K, D, V]`.
    tables: Array,
    heads: Array,
    /// The heads' norm weights `[K, D]`, f32.
    head_norms: Array,
    /// The model's `codebook_embedding` `[K * V, hidden]`: a frame's input row.
    frame_table: Linear,
    codebooks: u32,
    vocab: u32,
    /// The end-of-speech code.
    eos: u32,
    dim: c_int,
    eps: f32,
    schedule: []const u32,
    s: mlx.mlx_stream,

    pub fn load(ctx: *const abi.LoadCtx, block_json: []const u8, owned: *reader.Owned, codebooks: u32, eos: u32, passes: []const u32, eps: f32, theta: f32, layers: []const bool, hidden: u32) !?Depth {
        var r = try reader.Reader.open(ctx, block_json, "depth_decoder");
        defer r.close();
        if (!r.has("depth_decoder.mask_embedding")) return null;
        const mask = try r.need("depth_decoder.mask_embedding");
        try owned.own(mask);
        const dim: u32 = @intCast(ops.shape(mask)[0]);
        var g = Graph.init(gpa, r.s);
        defer g.deinit();
        var tables: [max_codebooks]Array = undefined;
        var norms: [max_codebooks]Array = undefined;
        var nb: [160]u8 = undefined;
        for (0..codebooks) |i| {
            tables[i] = try g.adopt(try r.need(try std.fmt.bufPrint(&nb, "depth_decoder.depth_embeddings.{d}.embedding.weight", .{i})));
            norms[i] = try g.astype(try g.adopt(try r.need(try std.fmt.bufPrint(&nb, "depth_decoder.depth_embeddings.{d}.embedding_norm.weight", .{i}))), .float32);
        }
        const vocab: u32 = @intCast(ops.shape(tables[0])[0]);
        const stacked = try stack(&g, tables[0..codebooks]);
        const flat = g.escape(try g.contiguous(try g.reshape(stacked, &.{ -1, @intCast(dim) })));
        try owned.own(flat);
        const heads = g.escape(try g.contiguous(try g.transpose(stacked, &.{ 0, 2, 1 })));
        try owned.own(heads);
        const head_norms = g.escape(try stack(&g, norms[0..codebooks]));
        try owned.own(head_norms);
        for ([_]Array{ flat, heads, head_norms }) |a| try mlx.check(mlx.mlx_array_eval(a));

        var linear = try r.proj("depth_decoder.depth_linear", hidden);
        linear.bias = try r.need("depth_decoder.depth_linear.bias");
        try owned.linear(linear);
        const frame_table = try r.table("codebook_embedding", hidden);
        try owned.linear(frame_table);
        return .{
            .linear = linear,
            .mask = mask,
            .trunk = try trunk_mod.Trunk.load(&r, owned, "depth_decoder.model", layers, dim, eps, theta, .bidirectional),
            .tables = flat,
            .heads = heads,
            .head_norms = head_norms,
            .frame_table = frame_table,
            .codebooks = codebooks,
            .vocab = vocab,
            .eos = eos,
            .dim = @intCast(dim),
            .eps = eps,
            .schedule = passes,
            .s = r.s,
        };
    }

    pub fn deinit(d: *Depth) void {
        d.trunk.deinit();
    }

    /// One frame's codes into `out` (`codebooks` of them) from the trunk's hidden row `[1, hidden]` (final-normed);
    /// `history` holds earlier frames' codes, frame after frame, for the per-codebook repetition penalty.
    pub fn frame(d: *const Depth, hidden: Array, sp: Sampling, history: []const u32, out: []u32) !void {
        const k: c_int = @intCast(d.codebooks);
        var g = Graph.init(gpa, d.s);
        defer g.deinit();
        const x0 = try g.reshape(try d.linear.forward(&g, hidden), &.{ 1, k, d.dim });
        var offs: [max_codebooks]i32 = undefined;
        for (offs[0..d.codebooks], 0..) |*o, i| o.* = @intCast(i * d.vocab);
        const offsets = try g.fromSlice(i32, offs[0..d.codebooks], &.{k});
        const hist: ?Array = if (sp.repetition_penalty != 1.0 and history.len >= d.codebooks) blk: {
            const n: c_int = @intCast(history.len / d.codebooks);
            const h = try g.fromSlice(u32, history, &.{ n, k });
            break :blk try g.astype(try g.transpose(h, &.{ 1, 0 }), .int32);
        } else null;
        var codes = try g.zeros(&.{k}, .int32);
        var revealed: [max_codebooks]bool = @splat(false);
        var start: usize = 0;
        for (d.schedule, 0..) |size, pass| {
            var pg = g.scope();
            defer pg.deinit();
            const shown = try pg.fromSlice(bool, revealed[0..d.codebooks], &.{ k, 1 });
            const embeds = try pg.take(d.tables, try pg.add(codes, offsets), 0);
            const mixed = try pg.where(shown, embeds, try pg.astype(d.mask, mlx.mlx_array_dtype(.{ .ctx = embeds.ctx })));
            const h = try d.trunk.forward(&pg, try pg.add(x0, try pg.expandDims(mixed, 0)), 0);
            const logits = try d.headLogits(&pg, h);
            const sampled = try sample(&pg, logits, sp, hist, pass);
            var pick: [max_codebooks]bool = @splat(false);
            for (sp.order[start .. start + size]) |c| pick[c] = true;
            start += size;
            const picked = try pg.fromSlice(bool, pick[0..d.codebooks], &.{k});
            codes = try g.keep(try pg.where(picked, sampled, codes));
            for (0..d.codebooks) |c| revealed[c] = revealed[c] or pick[c];
        }
        const host = try g.astype(codes, .uint32);
        try mlx.check(mlx.mlx_array_eval(host));
        const data = mlx.mlx_array_data_uint32(host) orelse return error.NoData;
        @memcpy(out[0..d.codebooks], data[0..d.codebooks]);
    }

    /// Every slot's logits `[K, V]`: its row normed in f32 by its head's weight, rounded once, against its table.
    fn headLogits(d: *const Depth, g: *Graph, h: Array) !Array {
        const dt = mlx.mlx_array_dtype(.{ .ctx = h.ctx });
        const x = try g.astype(try g.reshape(h, &.{ @intCast(d.codebooks), d.dim }), .float32);
        const ms = try g.mean(try g.square(x), -1, true);
        const normed = try g.astype(try g.mul(try g.mul(x, try g.rsqrt(try g.addScalar(ms, d.eps))), d.head_norms), dt);
        return g.reshape(try g.matmul(try g.expandDims(normed, 1), d.heads), &.{ @intCast(d.codebooks), @intCast(d.vocab) });
    }

    /// Whole frames' input rows `[1, frames, hidden]`: each the sum of its codes' rows of `codebook_embedding`.
    pub fn frameRows(d: *const Depth, codes: []const u32) !mlx.mlx_array {
        if (codes.len == 0 or codes.len % d.codebooks != 0) return error.PartialFrame;
        var g = Graph.init(gpa, d.s);
        defer g.deinit();
        const ids = try gpa.alloc(i32, codes.len);
        defer gpa.free(ids);
        for (codes, ids, 0..) |c, *id, i| {
            if (c >= d.vocab) return error.CodeOutOfRange;
            id.* = @intCast(c + @as(u32, @intCast(i % d.codebooks)) * d.vocab);
        }
        const frames: c_int = @intCast(codes.len / d.codebooks);
        const rows = try bases.blocks.embed.gatherRows(&g, &d.frame_table, try g.fromSlice(i32, ids, &.{@intCast(codes.len)}), .bfloat16);
        const summed = try g.sum(try g.reshape(rows, &.{ frames, @intCast(d.codebooks), -1 }), 1, false);
        return g.escape(try g.reshape(summed, &.{ 1, frames, -1 }));
    }
};

/// `xs` stacked along a new leading axis.
fn stack(g: *Graph, xs: []const Array) !Array {
    var rows: [max_codebooks]Array = undefined;
    for (xs, 0..) |x, i| rows[i] = try g.expandDims(x, 0);
    return g.concat(rows[0..xs.len], 0);
}

/// mlx-vlm's `sample` per row of `logits` `[K, V]`: the repetition penalty over each row's own history (greedy too),
/// then argmax at temperature 0, else temperature, log-softmax, top-k and a categorical draw.
fn sample(g: *Graph, logits: Array, sp: Sampling, history: ?Array, pass: usize) !Array {
    var scores = try g.astype(logits, .float32);
    if (history) |ids| {
        const picked = try g.takeAlong(scores, ids, 1);
        const neg = try g.less(picked, try g.scalar(0, .float32));
        const penalized = try g.where(neg, try g.mulScalar(picked, sp.repetition_penalty), try g.mulScalar(picked, 1.0 / sp.repetition_penalty));
        scores = try g.call(mlx.mlx_put_along_axis, .{ scores, ids, penalized, @as(c_int, 1) });
    }
    if (sp.temperature == 0) return g.astype(try g.argmax(scores, -1, false), .int32);
    scores = try g.mulScalar(scores, 1.0 / sp.temperature);
    scores = try g.sub(scores, try g.call(mlx.mlx_logsumexp_axis, .{ scores, @as(c_int, -1), true }));
    const vocab: u32 = @intCast(ops.shape(scores)[1]);
    if (sp.top_k > 0 and sp.top_k < vocab) {
        const drop = try g.sliceAxis(try g.call(mlx.mlx_argpartition_axis, .{ try g.neg(scores), @as(c_int, @intCast(sp.top_k - 1)), @as(c_int, -1) }), 1, @intCast(sp.top_k), @intCast(vocab));
        const ninf = try g.broadcast(try g.scalar(-std.math.inf(f32), .float32), ops.shape(drop));
        scores = try g.call(mlx.mlx_put_along_axis, .{ scores, drop, ninf, @as(c_int, 1) });
    }
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, sp.key +% pass));
    const k = try g.adopt(key);
    return g.astype(try g.call(mlx.mlx_random_categorical, .{ scores, @as(c_int, -1), k }), .int32);
}

/// The reveal order of a frame left to chance (the reference draws one permutation per frame).
pub fn randomOrder(buf: []u32, key: u64) []u32 {
    for (buf, 0..) |*o, i| o.* = @intCast(i);
    var rng = std.Random.DefaultPrng.init(key);
    rng.random().shuffle(u32, buf);
    return buf;
}

/// A frame with any end-of-speech code is all end-of-speech.
pub fn normalizeEos(codes: []u32, eos: u32) bool {
    for (codes) |c| if (c == eos) {
        @memset(codes, eos);
        return true;
    };
    return false;
}

/// The MaskGIT schedule: `steps` passes over `codebooks`, the remainder on the last.
pub fn schedule(buf: []u32, codebooks: u32, steps: u32) []u32 {
    for (buf[0..steps]) |*s| s.* = codebooks / steps;
    buf[steps - 1] += codebooks % steps;
    return buf[0..steps];
}

// ── tests ──

test "lfm3 depth: a frame holding one end-of-speech code is all end-of-speech; the schedule splits codebooks" {
    var f = [_]u32{ 3, 2016, 5 };
    try std.testing.expect(normalizeEos(&f, 2016));
    try std.testing.expectEqualSlices(u32, &.{ 2016, 2016, 2016 }, &f);
    var g = [_]u32{ 3, 4, 5 };
    try std.testing.expect(!normalizeEos(&g, 2016));
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 4, 4, 4 }, schedule(&buf, 12, 3));
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 2 }, schedule(&buf, 4, 3));
}
