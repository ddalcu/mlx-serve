//! CLIP-L text tower for FLUX.1 (`text_encoder`) → pooled 768-d vector. This is
//! the vector conditioning fed to the DiT's `text_embedder`. Standard HF
//! CLIPTextModel: learned token+position embeddings, causal self-attention,
//! quick-gelu MLP, final layer norm; the pooled output is the hidden state at
//! the EOS position (argmax of the token ids). See docs/reference.md (FLUX.1).
//!
//! Keys are HF-standard (`text_model.embeddings.*`, `text_model.encoder.layers.N.*`,
//! `text_model.final_layer_norm`); linears + embeddings are affine quantized.
//! (A twin CLIP-L already exists in sdxl_clip.zig; we keep a flux-local copy to
//! avoid coupling to the in-flight SDXL work and to control the pooled path.)

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const fc = @import("flux_common.zig");

const QLinear = fc.QLinear;
const S = fc.S;
const Weights = model_mod.Weights;
const addA = fc.addA;
const mulA = fc.mulA;
const reshape = fc.reshape;
const transpose = fc.transpose;
const astype = fc.astype;

pub const ClipConfig = struct {
    hidden: u32 = 768,
    layers: u32 = 12,
    heads: u32 = 12,
    head_dim: u32 = 64,
    eps: f32 = 1e-5,
};

/// LayerNorm with weight+bias (CLIP uses full LayerNorm, not RMS).
fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    // mean over last axis
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, xf, -1, true, s));
    const xc = try fc.subA(xf, mean, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer _ = mlx.mlx_array_free(norm);
    const wf = try astype(w, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(b, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const sc = try mulA(norm, wf, s);
    defer _ = mlx.mlx_array_free(sc);
    const out = try addA(sc, bf, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}

/// quick-gelu: x · sigmoid(1.702·x).
fn quickGelu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(1.702);
    defer _ = mlx.mlx_array_free(c);
    const cx = try mulA(x, c, s);
    defer _ = mlx.mlx_array_free(cx);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, cx, s));
    return mulA(x, sig, s);
}

const ClipLayer = struct {
    ln1w: mlx.mlx_array,
    ln1b: mlx.mlx_array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    ln2w: mlx.mlx_array,
    ln2b: mlx.mlx_array,
    fc1: QLinear,
    fc2: QLinear,
    fn deinit(self: *ClipLayer) void {
        inline for (.{ "ln1w", "ln1b", "ln2w", "ln2b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

pub const ClipEncoder = struct {
    cfg: ClipConfig,
    allocator: std.mem.Allocator,
    s: S,
    token_embed: mlx.mlx_array, // dequant [vocab, hidden] bf16
    pos_embed: mlx.mlx_array, // dequant [max_pos, hidden] bf16
    layers: []ClipLayer,
    final_w: mlx.mlx_array,
    final_b: mlx.mlx_array,

    pub fn deinit(self: *ClipEncoder) void {
        _ = mlx.mlx_array_free(self.token_embed);
        _ = mlx.mlx_array_free(self.pos_embed);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.final_w);
        _ = mlx.mlx_array_free(self.final_b);
    }

    /// Encode token ids [seq] → pooled hidden at the EOS position [1, hidden] (bf16).
    pub fn encodePooled(self: *ClipEncoder, ids: []const i32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const seq: c_int = @intCast(ids.len);
        const H: c_int = @intCast(c.hidden);
        const heads: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.head_dim);

        // token + position embeds
        const id_shape = [_]c_int{seq};
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var tok = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_take_axis(&tok, self.token_embed, id_arr, 0, s));
        // position ids 0..seq
        const pos_ids = try self.allocator.alloc(i32, ids.len);
        defer self.allocator.free(pos_ids);
        for (0..ids.len) |i| pos_ids[i] = @intCast(i);
        const pos_arr = mlx.mlx_array_new_data(pos_ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(pos_arr);
        var pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos);
        try mlx.check(mlx.mlx_take_axis(&pos, self.pos_embed, pos_arr, 0, s));
        const summed = try addA(tok, pos, s);
        defer _ = mlx.mlx_array_free(summed);
        var x = try reshape(summed, &[_]c_int{ 1, seq, H }, s);
        errdefer _ = mlx.mlx_array_free(x);

        // causal mask [1,1,seq,seq]
        const mask = try self.causalMask(@intCast(seq));
        defer _ = mlx.mlx_array_free(mask);

        for (self.layers) |*layer| {
            const nx = try self.layerForward(x, layer, mask, seq, heads, hd, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        const fx = try layerNorm(x, self.final_w, self.final_b, c.eps, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(fx);

        // pooled = hidden at eos position = argmax(ids)
        var eos_pos: usize = 0;
        var max_id: i32 = ids[0];
        for (ids, 0..) |id, i| {
            if (id > max_id) {
                max_id = id;
                eos_pos = i;
            }
        }
        return fc.slice3(fx, 1, @intCast(eos_pos), @intCast(eos_pos + 1), s); // [1,1,hidden]
    }

    fn causalMask(self: *ClipEncoder, seq: c_int) !mlx.mlx_array {
        const n: usize = @intCast(seq);
        const buf = try self.allocator.alloc(f32, n * n);
        defer self.allocator.free(buf);
        const neg = -std.math.inf(f32);
        for (0..n) |i| {
            for (0..n) |j| buf[i * n + j] = if (j > i) neg else 0.0;
        }
        const shape = [_]c_int{ 1, 1, seq, seq };
        return mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    }

    fn layerForward(self: *ClipEncoder, x: mlx.mlx_array, layer: *const ClipLayer, mask: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        const eps = self.cfg.eps;
        const xn = try layerNorm(x, layer.ln1w, layer.ln1b, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try layer.q.forward(xn, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try layer.k.forward(xn, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try layer.v.forward(xn, s);
        defer _ = mlx.mlx_array_free(v);
        const qt = try toHeads(q, seq, heads, hd, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = try toHeads(k, seq, heads, hd, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = try toHeads(v, seq, heads, hd, s);
        defer _ = mlx.mlx_array_free(vt);
        const qf = try astype(qt, .float32, s);
        defer _ = mlx.mlx_array_free(qf);
        const kf = try astype(kt, .float32, s);
        defer _ = mlx.mlx_array_free(kf);
        const vf = try astype(vt, .float32, s);
        defer _ = mlx.mlx_array_free(vf);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.cfg.head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_sink = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qf, kf, vf, scale, "array", mask, null_sink, false, s));
        const attn_bf = try astype(attn, .bfloat16, s);
        defer _ = mlx.mlx_array_free(attn_bf);
        const at = try transpose(attn_bf, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try layer.o.forward(af, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(x, o, s);
        defer _ = mlx.mlx_array_free(h1);
        // MLP
        const hn = try layerNorm(h1, layer.ln2w, layer.ln2b, eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const f1 = try layer.fc1.forward(hn, s);
        defer _ = mlx.mlx_array_free(f1);
        const ga = try quickGelu(f1, s);
        defer _ = mlx.mlx_array_free(ga);
        const f2 = try layer.fc2.forward(ga, s);
        defer _ = mlx.mlx_array_free(f2);
        return addA(h1, f2, s);
    }
};

fn toHeads(x: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const r = try reshape(x, &[_]c_int{ 1, seq, heads, hd }, s);
    defer _ = mlx.mlx_array_free(r);
    return transpose(r, &[_]c_int{ 0, 2, 1, 3 }, s);
}

fn dequantWeight(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    var ql = try QLinear.load(w, a, prefix);
    defer ql.deinit();
    return fc.dequantTable(ql.w, ql.scales, ql.biases, ql.bits, ql.group_size, s);
}

pub fn loadEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !ClipEncoder {
    const dir = try fc.fmtKey(allocator, "{s}/text_encoder", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();

    var enc: ClipEncoder = undefined;
    enc.allocator = allocator;
    enc.s = s;
    enc.token_embed = try dequantWeight(&w, allocator, "text_model.embeddings.token_embedding", s);
    enc.pos_embed = try dequantWeight(&w, allocator, "text_model.embeddings.position_embedding", s);
    const hidden: u32 = @intCast(mlx.getShape(enc.token_embed)[1]);
    enc.cfg = .{
        .hidden = hidden,
        .layers = fc.countIndexed(&w, allocator, "text_model.encoder.layers.{d}.layer_norm1.weight"),
        .head_dim = 64,
        .heads = hidden / 64,
    };
    if (enc.cfg.layers == 0) enc.cfg.layers = 12;
    log.info("[clip] encoder: hidden={d} layers={d} heads={d}\n", .{ enc.cfg.hidden, enc.cfg.layers, enc.cfg.heads });

    enc.layers = try allocator.alloc(ClipLayer, enc.cfg.layers);
    errdefer allocator.free(enc.layers);
    var loaded: usize = 0;
    errdefer for (enc.layers[0..loaded]) |*l| l.deinit();
    for (0..enc.cfg.layers) |li| {
        enc.layers[li] = try loadLayer(&w, allocator, li);
        loaded += 1;
    }
    enc.final_w = try fc.ownWeight(&w, "text_model.final_layer_norm.weight");
    enc.final_b = try fc.ownWeight(&w, "text_model.final_layer_norm.bias");
    return enc;
}

fn loadLayer(w: *const Weights, a: std.mem.Allocator, li: usize) !ClipLayer {
    const p = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}", .{li});
    defer a.free(p);
    const k = struct {
        fn norm(ww: *const Weights, aa: std.mem.Allocator, pfx: []const u8, sub: []const u8) !mlx.mlx_array {
            const kk = try std.fmt.allocPrint(aa, "{s}.{s}", .{ pfx, sub });
            defer aa.free(kk);
            return fc.ownWeight(ww, kk);
        }
        fn ql(ww: *const Weights, aa: std.mem.Allocator, pfx: []const u8, sub: []const u8) !QLinear {
            const kk = try std.fmt.allocPrint(aa, "{s}.{s}", .{ pfx, sub });
            defer aa.free(kk);
            return QLinear.load(ww, aa, kk);
        }
    };
    return .{
        .ln1w = try k.norm(w, a, p, "layer_norm1.weight"),
        .ln1b = try k.norm(w, a, p, "layer_norm1.bias"),
        .q = try k.ql(w, a, p, "self_attn.q_proj"),
        .k = try k.ql(w, a, p, "self_attn.k_proj"),
        .v = try k.ql(w, a, p, "self_attn.v_proj"),
        .o = try k.ql(w, a, p, "self_attn.out_proj"),
        .ln2w = try k.norm(w, a, p, "layer_norm2.weight"),
        .ln2b = try k.norm(w, a, p, "layer_norm2.bias"),
        .fc1 = try k.ql(w, a, p, "mlp.fc1"),
        .fc2 = try k.ql(w, a, p, "mlp.fc2"),
    };
}
