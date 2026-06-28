//! Pure-MLX NSFW image classifier — a faithful port of the HF
//! `ViTForImageClassification` used by `Falconsai/nsfw_image_detection`
//! (ViT-base/16-224, Apache-2.0, 2 classes: 0=normal, 1=nsfw).
//!
//! Used as the image-generation content filter (Krea 2 Community License §4.2):
//! the server runs every generated image through it and redacts outputs that
//! score above a threshold. Weights are the ORIGINAL Falconsai safetensors,
//! loaded directly (f32) — no conversion/hosting; this file does the conv/linear
//! layout transposes at load. Auto-downloaded to `~/.mlx-serve/models`; missing →
//! the caller fails OPEN (warns + lets the image through).

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── mlx helpers (file-local; mirror flux/krea) ──

inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    return o;
}
inline fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, eps, s));
    return o;
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// exact GELU: 0.5*x*(1+erf(x/√2))
fn gelu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const xs = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(xs);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, xs, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const ep1 = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(ep1);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, ep1, s);
}

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[nsfw] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingNsfwWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}

// ── Config (ViT-base/16-224) ──

const HIDDEN: c_int = 768;
const LAYERS: usize = 12;
const HEADS: c_int = 12;
const HEAD_DIM: c_int = 64; // 768/12
const PATCH: c_int = 16;
const IMG: usize = 224;
const GRID: c_int = 14; // 224/16
const NPATCH: c_int = 196; // 14*14
const SEQ: c_int = 197; // +CLS
const LN_EPS: f32 = 1e-12;
pub const NSFW_THRESHOLD: f32 = 0.5; // P(nsfw) above this → flagged

/// nn.Linear stored [out,in]; pre-transposed to [in,out] at load + bias.
const Linear = struct {
    w_t: mlx.mlx_array,
    bias: mlx.mlx_array,
    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !Linear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const bk = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(bk);
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        return .{ .w_t = try contig(t, s), .bias = try ownWeight(w, bk) };
    }
    fn deinit(self: *Linear) void {
        _ = mlx.mlx_array_free(self.w_t);
        _ = mlx.mlx_array_free(self.bias);
    }
    fn forward(self: *const Linear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const o = try matmul(x, self.w_t, s);
        defer _ = mlx.mlx_array_free(o);
        return addA(o, self.bias, s);
    }
};

const LayerNormW = struct {
    w: mlx.mlx_array,
    b: mlx.mlx_array,
    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) !LayerNormW {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const bk = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(bk);
        return .{ .w = try ownWeight(w, wk), .b = try ownWeight(w, bk) };
    }
    fn deinit(self: *LayerNormW) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.b);
    }
};

const Layer = struct {
    ln_before: LayerNormW,
    q: Linear,
    k: Linear,
    v: Linear,
    attn_out: Linear,
    ln_after: LayerNormW,
    inter: Linear,
    out: Linear,
    fn deinit(self: *Layer) void {
        self.ln_before.deinit();
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.attn_out.deinit();
        self.ln_after.deinit();
        self.inter.deinit();
        self.out.deinit();
    }
};

pub const Classifier = struct {
    allocator: std.mem.Allocator,
    s: S,
    patch_w: mlx.mlx_array, // conv [768,16,16,3] OHWI
    patch_b: mlx.mlx_array, // [768]
    cls_token: mlx.mlx_array, // [1,1,768]
    pos_embed: mlx.mlx_array, // [1,197,768]
    layers: [LAYERS]Layer,
    final_ln: LayerNormW,
    head: Linear, // [2,768] -> logits

    pub fn deinit(self: *Classifier) void {
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.patch_b);
        _ = mlx.mlx_array_free(self.cls_token);
        _ = mlx.mlx_array_free(self.pos_embed);
        for (&self.layers) |*l| l.deinit();
        self.final_ln.deinit();
        self.head.deinit();
    }

    /// pixel_values [1,3,224,224] f32 (already resized + normalized to [-1,1]) →
    /// logits [1,2] f32.
    pub fn forward(self: *Classifier, pixels_nchw: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        // patch embed: NCHW → NHWC, conv (stride 16), → [1,196,768]
        const nhwc = try transpose(pixels_nchw, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc);
        const nhwc_c = try contig(nhwc, s);
        defer _ = mlx.mlx_array_free(nhwc_c);
        var conv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv2d(&conv, nhwc_c, self.patch_w, PATCH, PATCH, 0, 0, 1, 1, 1, s));
        defer _ = mlx.mlx_array_free(conv); // [1,14,14,768]
        const flat = try reshape(conv, &[_]c_int{ 1, NPATCH, HIDDEN }, s);
        defer _ = mlx.mlx_array_free(flat);
        const pb = try reshape(self.patch_b, &[_]c_int{ 1, 1, HIDDEN }, s);
        defer _ = mlx.mlx_array_free(pb);
        const patches = try addA(flat, pb, s);
        defer _ = mlx.mlx_array_free(patches);
        // prepend CLS, add position embeddings
        var x = try concat(&[_]mlx.mlx_array{ self.cls_token, patches }, 1, s); // [1,197,768]
        {
            const nx = try addA(x, self.pos_embed, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        // encoder layers (pre-norm)
        for (&self.layers) |*layer| {
            const nx = try layerForward(x, layer, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        // final LN → CLS token → classifier head
        const ln = try layerNorm(x, self.final_ln.w, self.final_ln.b, LN_EPS, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(ln);
        const cls = blk: {
            var lo = [_]c_int{ 0, 0, 0 };
            var hi = [_]c_int{ 1, 1, HIDDEN };
            const st = [_]c_int{ 1, 1, 1 };
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&o, ln, &lo, 3, &hi, 3, &st, 3, s));
            break :blk o;
        }; // [1,1,768]
        defer _ = mlx.mlx_array_free(cls);
        const cls2 = try reshape(cls, &[_]c_int{ 1, HIDDEN }, s);
        defer _ = mlx.mlx_array_free(cls2);
        return self.head.forward(cls2, s); // [1,2]
    }

    fn layerForward(x: mlx.mlx_array, layer: *const Layer, s: S) !mlx.mlx_array {
        // attention (pre-norm)
        const h = try layerNorm(x, layer.ln_before.w, layer.ln_before.b, LN_EPS, s);
        defer _ = mlx.mlx_array_free(h);
        const q0 = try layer.q.forward(h, s);
        defer _ = mlx.mlx_array_free(q0);
        const k0 = try layer.k.forward(h, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try layer.v.forward(h, s);
        defer _ = mlx.mlx_array_free(v0);
        const qh = try splitHeads(q0, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k0, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v0, s);
        defer _ = mlx.mlx_array_free(vh);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(HEAD_DIM)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qh, kh, vh, scale, "", null_a, null_a, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s); // [1,197,12,64]
        defer _ = mlx.mlx_array_free(at);
        const merged = try reshape(at, &[_]c_int{ 1, SEQ, HIDDEN }, s);
        defer _ = mlx.mlx_array_free(merged);
        const ao = try layer.attn_out.forward(merged, s);
        defer _ = mlx.mlx_array_free(ao);
        const x1 = try addA(x, ao, s);
        defer _ = mlx.mlx_array_free(x1);
        // MLP (pre-norm)
        const h2 = try layerNorm(x1, layer.ln_after.w, layer.ln_after.b, LN_EPS, s);
        defer _ = mlx.mlx_array_free(h2);
        const inter = try layer.inter.forward(h2, s);
        defer _ = mlx.mlx_array_free(inter);
        const act = try gelu(inter, s);
        defer _ = mlx.mlx_array_free(act);
        const mlp = try layer.out.forward(act, s);
        defer _ = mlx.mlx_array_free(mlp);
        return addA(x1, mlp, s);
    }

    /// [1,197,768] → [1,12,197,64]
    fn splitHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const r = try reshape(x, &[_]c_int{ 1, SEQ, HEADS, HEAD_DIM }, s);
        defer _ = mlx.mlx_array_free(r);
        return transpose(r, &[_]c_int{ 0, 2, 1, 3 }, s);
    }

    /// Generated image [1,3,H,W] f32 in [0,1] → P(nsfw) in [0,1].
    pub fn classify(self: *Classifier, image_f32: mlx.mlx_array) !f32 {
        const s = self.s;
        const pixels = try preprocess(self.allocator, image_f32, s);
        defer _ = mlx.mlx_array_free(pixels);
        const logits = try self.forward(pixels);
        defer _ = mlx.mlx_array_free(logits);
        var sm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sm);
        try mlx.check(mlx.mlx_softmax_axis(&sm, logits, -1, true, s));
        _ = mlx.mlx_array_eval(sm);
        const d = mlx.mlx_array_data_float32(sm) orelse return error.NoData;
        return d[1]; // P(nsfw)
    }
};

/// Bilinear-resize an image [1,3,H,W] f32 [0,1] to [1,3,224,224] and normalize to
/// [-1,1] (ViT processor: rescale 1/255 already applied since input is [0,1],
/// then (x-0.5)/0.5). Resize runs on CPU (align_corners=false, PIL convention).
fn preprocess(allocator: std.mem.Allocator, image_f32: mlx.mlx_array, s: S) !mlx.mlx_array {
    const cf = try contig(image_f32, s);
    defer _ = mlx.mlx_array_free(cf);
    _ = mlx.mlx_array_eval(cf);
    const sh = mlx.getShape(cf); // [1,3,H,W]
    const H: usize = @intCast(sh[2]);
    const W: usize = @intCast(sh[3]);
    const src = mlx.mlx_array_data_float32(cf) orelse return error.NoData;
    const out = try allocator.alloc(f32, 3 * IMG * IMG);
    defer allocator.free(out);
    const plane_src = H * W;
    const plane_dst = IMG * IMG;
    const sy: f32 = @as(f32, @floatFromInt(H)) / @as(f32, @floatFromInt(IMG));
    const sx: f32 = @as(f32, @floatFromInt(W)) / @as(f32, @floatFromInt(IMG));
    for (0..IMG) |oy| {
        var fy = (@as(f32, @floatFromInt(oy)) + 0.5) * sy - 0.5;
        if (fy < 0) fy = 0;
        const y0f = @floor(fy);
        var y0: usize = @intFromFloat(y0f);
        if (y0 > H - 1) y0 = H - 1;
        var y1 = y0 + 1;
        if (y1 > H - 1) y1 = H - 1;
        const wy = fy - y0f;
        for (0..IMG) |ox| {
            var fx = (@as(f32, @floatFromInt(ox)) + 0.5) * sx - 0.5;
            if (fx < 0) fx = 0;
            const x0f = @floor(fx);
            var x0: usize = @intFromFloat(x0f);
            if (x0 > W - 1) x0 = W - 1;
            var x1 = x0 + 1;
            if (x1 > W - 1) x1 = W - 1;
            const wx = fx - x0f;
            for (0..3) |c| {
                const base = c * plane_src;
                const v00 = src[base + y0 * W + x0];
                const v01 = src[base + y0 * W + x1];
                const v10 = src[base + y1 * W + x0];
                const v11 = src[base + y1 * W + x1];
                const top = v00 + (v01 - v00) * wx;
                const bot = v10 + (v11 - v10) * wx;
                const val = top + (bot - top) * wy;
                out[c * plane_dst + oy * IMG + ox] = val * 2.0 - 1.0; // [0,1]→[-1,1]
            }
        }
    }
    const shape = [_]c_int{ 1, 3, IMG, IMG };
    return mlx.mlx_array_new_data(out.ptr, &shape, 4, .float32);
}

/// Load the Falconsai ViT from `model_dir` (its original f32 safetensors).
pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Classifier {
    var w = try model_mod.loadWeights(io, allocator, model_dir);
    defer w.deinit();
    var c: Classifier = undefined;
    c.allocator = allocator;
    c.s = mlx.mlx_default_gpu_stream_new();
    const s = c.s;

    // patch conv: PyTorch [768,3,16,16] (OIHW) → MLX [768,16,16,3] (OHWI)
    {
        const raw = try ownWeight(&w, "vit.embeddings.patch_embeddings.projection.weight");
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(t);
        c.patch_w = try contig(t, s);
    }
    c.patch_b = try ownWeight(&w, "vit.embeddings.patch_embeddings.projection.bias");
    c.cls_token = try ownWeight(&w, "vit.embeddings.cls_token");
    c.pos_embed = try ownWeight(&w, "vit.embeddings.position_embeddings");

    for (&c.layers, 0..) |*layer, i| {
        const lb = try fmtKey(allocator, "vit.encoder.layer.{d}.layernorm_before", .{i});
        defer allocator.free(lb);
        const la = try fmtKey(allocator, "vit.encoder.layer.{d}.layernorm_after", .{i});
        defer allocator.free(la);
        const qp = try fmtKey(allocator, "vit.encoder.layer.{d}.attention.attention.query", .{i});
        defer allocator.free(qp);
        const kp = try fmtKey(allocator, "vit.encoder.layer.{d}.attention.attention.key", .{i});
        defer allocator.free(kp);
        const vp = try fmtKey(allocator, "vit.encoder.layer.{d}.attention.attention.value", .{i});
        defer allocator.free(vp);
        const op = try fmtKey(allocator, "vit.encoder.layer.{d}.attention.output.dense", .{i});
        defer allocator.free(op);
        const ip = try fmtKey(allocator, "vit.encoder.layer.{d}.intermediate.dense", .{i});
        defer allocator.free(ip);
        const dp = try fmtKey(allocator, "vit.encoder.layer.{d}.output.dense", .{i});
        defer allocator.free(dp);
        layer.* = .{
            .ln_before = try LayerNormW.load(&w, allocator, lb),
            .q = try Linear.load(&w, allocator, qp, s),
            .k = try Linear.load(&w, allocator, kp, s),
            .v = try Linear.load(&w, allocator, vp, s),
            .attn_out = try Linear.load(&w, allocator, op, s),
            .ln_after = try LayerNormW.load(&w, allocator, la),
            .inter = try Linear.load(&w, allocator, ip, s),
            .out = try Linear.load(&w, allocator, dp, s),
        };
    }
    c.final_ln = try LayerNormW.load(&w, allocator, "vit.layernorm");
    c.head = try Linear.load(&w, allocator, "classifier", s);
    return c;
}

// ── Tests ──

const testing = std.testing;

fn readF32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

// Parity oracle: feed the reference's preprocessed pixels, compare logits.
//   NSFW_TEST_MODEL=<Falconsai dir>, NSFW_PIXELS (f32 [1,3,224,224]), NSFW_LOGITS (f32 [2])
test "nsfw ViT reproduces reference logits" {
    const model_dir = std.mem.span(std.c.getenv("NSFW_TEST_MODEL") orelse return error.SkipZigTest);
    const px_p = std.mem.span(std.c.getenv("NSFW_PIXELS") orelse return error.SkipZigTest);
    const lg_p = std.mem.span(std.c.getenv("NSFW_LOGITS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const px = try readF32(io, a, px_p);
    defer a.free(px);
    const ref = try readF32(io, a, lg_p);
    defer a.free(ref);
    var clf = try load(io, a, model_dir);
    defer clf.deinit();
    const shape = [_]c_int{ 1, 3, IMG, IMG };
    const px_a = mlx.mlx_array_new_data(px.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(px_a);
    const logits = try clf.forward(px_a);
    defer _ = mlx.mlx_array_free(logits);
    _ = mlx.mlx_array_eval(logits);
    const d = mlx.mlx_array_data_float32(logits) orelse return error.NoData;
    std.debug.print("[nsfw-vit] logits=({d:.4},{d:.4}) ref=({d:.4},{d:.4})\n", .{ d[0], d[1], ref[0], ref[1] });
    try testing.expect(@abs(d[0] - ref[0]) < 0.05);
    try testing.expect(@abs(d[1] - ref[1]) < 0.05);
}

test "preprocess resizes + normalizes to [-1,1] at 224" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;
    // all-1.0 image [1,3,8,8] → resized all 1.0 → normalized to +1.0
    var buf: [3 * 8 * 8]f32 = undefined;
    @memset(&buf, 1.0);
    const sh = [_]c_int{ 1, 3, 8, 8 };
    const img = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    const px = try preprocess(a, img, s);
    defer _ = mlx.mlx_array_free(px);
    _ = mlx.mlx_array_eval(px);
    const osh = mlx.getShape(px);
    try testing.expectEqual(@as(c_int, 1), osh[0]);
    try testing.expectEqual(@as(c_int, 3), osh[1]);
    try testing.expectEqual(@as(c_int, 224), osh[2]);
    try testing.expectEqual(@as(c_int, 224), osh[3]);
    const d = mlx.mlx_array_data_float32(px).?;
    try testing.expect(@abs(d[0] - 1.0) < 1e-5); // 1.0 → 2*1-1 = 1.0
    try testing.expect(@abs(d[3 * 224 * 224 - 1] - 1.0) < 1e-5);
}
