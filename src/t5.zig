//! T5-XXL (google/t5-v1_1-xxl) encoder — the FLUX.1 secondary text encoder
//! (`text_encoder_2`). Ported to mlx-c FFI from the HF T5 reference. See
//! docs/reference.md (FLUX.1 section) and docs/gotchas/models-media.md.
//!
//! T5 specifics vs a decoder LM: bidirectional self-attention with a
//! **relative-position bias** (no RoPE), attention with **NO 1/√d scaling**,
//! **T5LayerNorm** (RMS with no mean-subtraction), and a **gated-GeLU** FF
//! (`gelu_new(wi_0·x) · (wi_1·x) → wo`). Following diffusers' FluxPipeline the
//! encoder runs with **no padding mask** — the full padded sequence is attended
//! (only the relative-position bias shapes attention).
//!
//! Weight naming is mflux's own (`shared`, `t5_blocks.N.attention.SelfAttention.*`,
//! `t5_blocks.N.ff.DenseReluDense.*`, `final_layer_norm`); linears are affine
//! quantized. The relative_attention_bias is stored per block but is identical
//! across blocks (HF shares block 0's) — we dequantize block 0's and reuse it.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const fc = @import("flux_common.zig");

const QLinear = fc.QLinear;
const S = fc.S;
const Weights = model_mod.Weights;
const matmul = fc.matmul;
const addA = fc.addA;
const mulA = fc.mulA;
const reshape = fc.reshape;
const transpose = fc.transpose;
const rms = fc.rms;
const astype = fc.astype;
const geluTanh = fc.geluTanh;

pub const T5Config = struct {
    d_model: u32 = 4096,
    d_ff: u32 = 10240,
    layers: u32 = 24,
    heads: u32 = 64,
    head_dim: u32 = 64,
    vocab: u32 = 32128,
    num_buckets: u32 = 32,
    max_distance: u32 = 128,
    eps: f32 = 1e-6,
};

const T5Block = struct {
    attn_ln: mlx.mlx_array, // attention.layer_norm.weight
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    ff_ln: mlx.mlx_array, // ff.layer_norm.weight
    wi_0: QLinear,
    wi_1: QLinear,
    wo: QLinear,

    fn deinit(self: *T5Block) void {
        _ = mlx.mlx_array_free(self.attn_ln);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.ff_ln);
        self.wi_0.deinit();
        self.wi_1.deinit();
        self.wo.deinit();
    }
};

pub const T5Encoder = struct {
    cfg: T5Config,
    allocator: std.mem.Allocator,
    s: S,
    embed_table: mlx.mlx_array, // dequantized [vocab, d_model] bf16
    blocks: []T5Block,
    final_ln: mlx.mlx_array,
    /// Dequantized block-0 relative_attention_bias table, f32 [num_buckets, heads].
    rel_bias_table: mlx.mlx_array,

    pub fn deinit(self: *T5Encoder) void {
        _ = mlx.mlx_array_free(self.embed_table);
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        _ = mlx.mlx_array_free(self.final_ln);
        _ = mlx.mlx_array_free(self.rel_bias_table);
    }

    /// Encode token ids [seq] (int32) → prompt embeds [1, seq, d_model] (bf16).
    /// No padding mask (FLUX convention): the whole padded sequence is attended.
    pub fn encode(self: *T5Encoder, ids: []const i32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const seq: c_int = @intCast(ids.len);
        const H: c_int = @intCast(c.d_model);
        const heads: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.head_dim);

        // Token embed lookup → [1, seq, d_model].
        const id_shape = [_]c_int{seq};
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, self.embed_table, id_arr, 0, s));
        var x = try reshape(taken, &[_]c_int{ 1, seq, H }, s);
        errdefer _ = mlx.mlx_array_free(x);

        // Position bias [1, heads, seq, seq] (f32), folded into the SDPA mask.
        const pos_bias = try self.buildPositionBias(@intCast(seq));
        defer _ = mlx.mlx_array_free(pos_bias);

        for (self.blocks) |*blk| {
            const nx = try self.blockForward(x, blk, pos_bias, seq, heads, hd, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        // final T5LayerNorm
        const fx = try rms(x, self.final_ln, c.eps, s);
        _ = mlx.mlx_array_free(x);
        return fx;
    }

    /// Build the relative-position bias [1, heads, seq, seq] (f32) from the
    /// dequantized bias table by gathering the per-(i,j) bucket rows.
    fn buildPositionBias(self: *T5Encoder, seq: usize) !mlx.mlx_array {
        const s = self.s;
        const a = self.allocator;
        const c = self.cfg;
        // Host bucket matrix [seq*seq] (row i, col j → bucket(j - i)).
        const buckets = try a.alloc(i32, seq * seq);
        defer a.free(buckets);
        for (0..seq) |i| {
            for (0..seq) |j| {
                const rel: i32 = @as(i32, @intCast(j)) - @as(i32, @intCast(i));
                buckets[i * seq + j] = @intCast(relBucket(rel, c.num_buckets, c.max_distance));
            }
        }
        const bsh = [_]c_int{@intCast(seq * seq)};
        const bidx = mlx.mlx_array_new_data(buckets.ptr, &bsh, 1, .int32);
        defer _ = mlx.mlx_array_free(bidx);
        // gather rows → [seq*seq, heads]
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_take_axis(&g, self.rel_bias_table, bidx, 0, s));
        // → [seq, seq, heads] → [heads, seq, seq] → [1, heads, seq, seq]
        const g3 = try reshape(g, &[_]c_int{ @intCast(seq), @intCast(seq), @intCast(c.heads) }, s);
        defer _ = mlx.mlx_array_free(g3);
        const gt = try transpose(g3, &[_]c_int{ 2, 0, 1 }, s);
        defer _ = mlx.mlx_array_free(gt);
        return reshape(gt, &[_]c_int{ 1, @intCast(c.heads), @intCast(seq), @intCast(seq) }, s);
    }

    fn blockForward(self: *T5Encoder, x: mlx.mlx_array, blk: *const T5Block, pos_bias: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        const eps = self.cfg.eps;
        // ── Self-attention ──
        const xn = try rms(x, blk.attn_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try blk.q.forward(xn, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try blk.k.forward(xn, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try blk.v.forward(xn, s);
        defer _ = mlx.mlx_array_free(v);
        // [1,seq,heads*hd] → [1,heads,seq,hd]
        const qt = try toHeads(q, seq, heads, hd, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = try toHeads(k, seq, heads, hd, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = try toHeads(v, seq, heads, hd, s);
        defer _ = mlx.mlx_array_free(vt);
        // SDPA in f32; T5 has NO 1/√d scaling (scale = 1.0); mask = position bias.
        const qf = try astype(qt, .float32, s);
        defer _ = mlx.mlx_array_free(qf);
        const kf = try astype(kt, .float32, s);
        defer _ = mlx.mlx_array_free(kf);
        const vf = try astype(vt, .float32, s);
        defer _ = mlx.mlx_array_free(vf);
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_sink = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qf, kf, vf, 1.0, "array", pos_bias, null_sink, false, s));
        const attn_bf = try astype(attn, .bfloat16, s);
        defer _ = mlx.mlx_array_free(attn_bf);
        const at = try transpose(attn_bf, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try blk.o.forward(af, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(x, o, s);
        defer _ = mlx.mlx_array_free(h1);
        // ── Gated-GeLU FF ──
        const hn = try rms(h1, blk.ff_ln, eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g0 = try blk.wi_0.forward(hn, s);
        defer _ = mlx.mlx_array_free(g0);
        const ga = try geluTanh(g0, s);
        defer _ = mlx.mlx_array_free(ga);
        const g1 = try blk.wi_1.forward(hn, s);
        defer _ = mlx.mlx_array_free(g1);
        const gg = try mulA(ga, g1, s);
        defer _ = mlx.mlx_array_free(gg);
        const dn = try blk.wo.forward(gg, s);
        defer _ = mlx.mlx_array_free(dn);
        return addA(h1, dn, s);
    }
};

/// [1, seq, heads*hd] → [1, heads, seq, hd].
fn toHeads(x: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const r = try reshape(x, &[_]c_int{ 1, seq, heads, hd }, s);
    defer _ = mlx.mlx_array_free(r);
    return transpose(r, &[_]c_int{ 0, 2, 1, 3 }, s);
}

/// HF T5 `_relative_position_bucket` (bidirectional). `rel = j - i`.
pub fn relBucket(rel: i32, num_buckets: u32, max_distance: u32) u32 {
    var ret: u32 = 0;
    const nb = num_buckets / 2; // 16
    var n: u32 = undefined;
    if (rel > 0) {
        ret += nb;
        n = @intCast(rel);
    } else {
        n = @intCast(-rel);
    }
    const max_exact = nb / 2; // 8
    if (n < max_exact) {
        ret += n;
    } else {
        const fn_ = @log(@as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(max_exact)));
        const fd = @log(@as(f32, @floatFromInt(max_distance)) / @as(f32, @floatFromInt(max_exact)));
        const scaled = fn_ / fd * @as(f32, @floatFromInt(nb - max_exact));
        var large: u32 = max_exact + @as(u32, @intFromFloat(scaled));
        if (large > nb - 1) large = nb - 1;
        ret += large;
    }
    return ret;
}

pub fn loadEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !T5Encoder {
    const dir = try fc.fmtKey(allocator, "{s}/text_encoder_2", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();

    var enc: T5Encoder = undefined;
    enc.allocator = allocator;
    enc.s = s;
    enc.cfg = .{
        .d_model = fc.rowsOf(&w, "t5_blocks.0.attention.layer_norm.weight"),
        .layers = fc.countIndexed(&w, allocator, "t5_blocks.{d}.attention.layer_norm.weight"),
        .d_ff = fc.rowsOf(&w, "t5_blocks.0.ff.DenseReluDense.wi_0.weight"),
        // heads = columns of the rel-bias table = its axis-0 dequant is [buckets, heads];
        // resolved after dequant below (kept at default 64 unless derivable).
    };
    if (enc.cfg.d_model == 0) enc.cfg.d_model = 4096;
    if (enc.cfg.layers == 0) enc.cfg.layers = 24;
    if (enc.cfg.d_ff == 0) enc.cfg.d_ff = 10240;

    // Embedding table (quantized) → bf16 [vocab, d_model].
    enc.embed_table = try dequantWeight(&w, allocator, "shared", s);
    enc.cfg.vocab = @intCast(mlx.getShape(enc.embed_table)[0]);

    // Block-0 relative-position bias table (quantized) → f32 [buckets, heads].
    const rb_bf = try dequantWeight(&w, allocator, "t5_blocks.0.attention.SelfAttention.relative_attention_bias", s);
    defer _ = mlx.mlx_array_free(rb_bf);
    enc.rel_bias_table = try astype(rb_bf, .float32, s);
    const rb_sh = mlx.getShape(enc.rel_bias_table);
    enc.cfg.num_buckets = @intCast(rb_sh[0]);
    enc.cfg.heads = @intCast(rb_sh[1]);
    enc.cfg.head_dim = @divTrunc(enc.cfg.d_model, enc.cfg.heads);

    log.info("[t5] encoder: d_model={d} layers={d} heads={d}x{d} d_ff={d} vocab={d} buckets={d}\n", .{
        enc.cfg.d_model, enc.cfg.layers, enc.cfg.heads, enc.cfg.head_dim, enc.cfg.d_ff, enc.cfg.vocab, enc.cfg.num_buckets,
    });

    enc.blocks = try allocator.alloc(T5Block, enc.cfg.layers);
    errdefer allocator.free(enc.blocks);
    var loaded: usize = 0;
    errdefer for (enc.blocks[0..loaded]) |*b| b.deinit();
    for (0..enc.cfg.layers) |li| {
        enc.blocks[li] = try loadBlock(&w, allocator, li);
        loaded += 1;
    }
    enc.final_ln = try fc.ownWeight(&w, "final_layer_norm.weight");
    return enc;
}

fn loadBlock(w: *const Weights, a: std.mem.Allocator, li: usize) !T5Block {
    const p = try std.fmt.allocPrint(a, "t5_blocks.{d}", .{li});
    defer a.free(p);
    const attn_ln = try keyed(w, a, "{s}.attention.layer_norm.weight", p);
    defer a.free(attn_ln.name);
    const ff_ln = try keyed(w, a, "{s}.ff.layer_norm.weight", p);
    defer a.free(ff_ln.name);
    return .{
        .attn_ln = try fc.ownWeight(w, attn_ln.name),
        .q = try qlin(w, a, "{s}.attention.SelfAttention.q", p),
        .k = try qlin(w, a, "{s}.attention.SelfAttention.k", p),
        .v = try qlin(w, a, "{s}.attention.SelfAttention.v", p),
        .o = try qlin(w, a, "{s}.attention.SelfAttention.o", p),
        .ff_ln = try fc.ownWeight(w, ff_ln.name),
        .wi_0 = try qlin(w, a, "{s}.ff.DenseReluDense.wi_0", p),
        .wi_1 = try qlin(w, a, "{s}.ff.DenseReluDense.wi_1", p),
        .wo = try qlin(w, a, "{s}.ff.DenseReluDense.wo", p),
    };
}

const Keyed = struct { name: []u8 };
fn keyed(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, prefix: []const u8) !Keyed {
    _ = w;
    return .{ .name = try std.fmt.allocPrint(a, fmt, .{prefix}) };
}
fn qlin(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, prefix: []const u8) !QLinear {
    const key = try std.fmt.allocPrint(a, fmt, .{prefix});
    defer a.free(key);
    return QLinear.load(w, a, key);
}

/// Dequantize a quantized weight table (`prefix.{weight,scales,biases}`) to bf16.
fn dequantWeight(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    var ql = try QLinear.load(w, a, prefix);
    defer ql.deinit();
    return fc.dequantTable(ql.w, ql.scales, ql.biases, ql.bits, ql.group_size, s);
}

// ── Tests ──
const testing = std.testing;

test "t5 relative position bucket matches HF reference" {
    // Bidirectional, num_buckets=32, max_distance=128.
    // Small exact region (|rel|<8): rel<=0 → bucket=|rel|; rel>0 → 16+rel.
    try testing.expectEqual(@as(u32, 0), relBucket(0, 32, 128));
    try testing.expectEqual(@as(u32, 1), relBucket(-1, 32, 128));
    try testing.expectEqual(@as(u32, 7), relBucket(-7, 32, 128));
    try testing.expectEqual(@as(u32, 17), relBucket(1, 32, 128));
    try testing.expectEqual(@as(u32, 23), relBucket(7, 32, 128));
    // Large region caps at num_buckets-1 (15 for the negative half, 31 positive).
    try testing.expectEqual(@as(u32, 15), relBucket(-127, 32, 128));
    try testing.expectEqual(@as(u32, 31), relBucket(127, 32, 128));
    // Boundary: |rel|=8 enters the log region (bucket 8 for negatives).
    try testing.expectEqual(@as(u32, 8), relBucket(-8, 32, 128));
    try testing.expectEqual(@as(u32, 24), relBucket(8, 32, 128));
}

// Encoder parity — env-gated on a real pack + an fp32 reference produced by
// tests/dump_t5_fixtures.py (HF T5EncoderModel, CPU fp32). Bar = cosine + RMS
// ratio (cosine alone can't see a scale error); the pack is 4-bit so an exact
// match is not expected. Skips without the env vars.
//   T5_TEST_MODEL = pack dir (…/flux-1-dev-mflux-q4)
//   T5_REF_IDS    = binary i32 token ids (no padding)
//   T5_REF_HIDDEN = binary f32 [seq, d_model] reference hidden states
test "t5 encoder parity vs fp32 reference" {
    const model_p = std.c.getenv("T5_TEST_MODEL") orelse return error.SkipZigTest;
    const ids_p = std.c.getenv("T5_REF_IDS") orelse return error.SkipZigTest;
    const ref_p = std.c.getenv("T5_REF_HIDDEN") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readBin(i32, io, a, std.mem.span(ids_p));
    defer a.free(ids);
    const ref = try readBin(f32, io, a, std.mem.span(ref_p));
    defer a.free(ref);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var enc = try loadEncoder(io, a, s, std.mem.span(model_p));
    defer enc.deinit();

    const out = try enc.encode(ids);
    defer _ = mlx.mlx_array_free(out);
    const outf = try astype(out, .float32, s);
    defer _ = mlx.mlx_array_free(outf);
    _ = mlx.mlx_array_eval(outf);
    const n: usize = @intCast(mlx.mlx_array_size(outf));
    try testing.expectEqual(ref.len, n);
    const data = mlx.mlx_array_data_float32(outf) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const x: f64 = data[i];
        const y: f64 = ref[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    const cos = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    const rms_ratio = std.math.sqrt(na / nb);
    std.debug.print("[t5-parity] n={d} cos={d:.5} rms_ratio={d:.4}\n", .{ n, cos, rms_ratio });
    try testing.expect(cos > 0.97);
    try testing.expect(rms_ratio > 0.85 and rms_ratio < 1.15);
}

fn readBin(comptime T: type, io: std.Io, a: std.mem.Allocator, path: []const u8) ![]T {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.reader(io, &buf);
    const bytes = try r.interface.allocRemaining(a, .limited(512 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / @sizeOf(T);
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * @sizeOf(T)]);
    return out;
}
