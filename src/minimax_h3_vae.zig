//! MiniMax H3 visual VAE — decoder side (f16t4d24).
//!
//! ASYMMETRIC: the encoder is a conv ResNet, but the DECODER is a 36-block
//! TRANSFORMER (2.6B params, hidden 2048) that goes straight from latent tokens
//! to pixels — each latent position emits one 3x4x16x16 block, which is where
//! the 16x spatial / 4x temporal compression comes from. There is no conv
//! upsampling stack; `decoder.*` is x_embedder, transformer_blocks, norm_out,
//! proj_out and two learned token buffers.
//!
//! Two structures here are semantic, not optimizations:
//!   * TEMPORAL CHUNKING. The VAE was trained on 17-frame clips = 5 latent
//!     tokens, so decode walks 5-token chunks with 2 tokens of overlap, drops
//!     `frame_pre_padding` (3) frames off each decoded chunk and cross-fades
//!     `frame_overlap` (5) frames. Decoding the whole clip in one pass is NOT
//!     equivalent.
//!   * SPATIAL TILING at 256 PIXELS. `create_token_ids` normalizes coordinates
//!     over whatever extent it is handed, so a tile's positions differ from the
//!     same region's positions in an untiled pass. At or below a 256-pixel
//!     extent `split_tiles` returns a single tile and the two agree; above it
//!     they do not, so `decode` REFUSES rather than silently producing
//!     off-distribution output (see `TilingUnsupported`).
//!
//! Ported from ComfyUI `comfy/ldm/minimax/vae.py`.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const h3 = @import("minimax_h3.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── Reference constants (vae.py MiniMaxH3VideoVAE.__init__) ─────────────────

/// Frames folded into one VAE clip. Drives every temporal constant below.
pub const CLIP_LENGTH: u32 = 17;
/// Temporal compression (prod of the encoder's time_down factors).
pub const VAE_RATIO_T: u32 = 4;
/// Spatial compression.
pub const VAE_RATIO: u32 = 16;
pub const TOKEN_DROP: u32 = 3;
/// Spatial tile extent, in PIXELS.
pub const TILE_SIZE: u32 = 256;

/// (-clip_length) % ratio_t = 3 — frames shaved off the front of each decode.
pub const FRAME_PRE_PADDING: u32 = (VAE_RATIO_T - (CLIP_LENGTH % VAE_RATIO_T)) % VAE_RATIO_T;
/// ceil(clip_length / ratio_t) = 5 latent tokens per chunk.
pub const TOKENS_CHUNK_SIZE: u32 = (CLIP_LENGTH + VAE_RATIO_T - 1) / VAE_RATIO_T;
/// (-token_drop) % tokens_chunk_size = 2.
pub const TOKEN_OVERLAP: u32 = (TOKENS_CHUNK_SIZE - (TOKEN_DROP % TOKENS_CHUNK_SIZE)) % TOKENS_CHUNK_SIZE;
/// max(overlap * ratio_t - pre_padding, 0) = 5 cross-faded frames.
pub const FRAME_OVERLAP: u32 = if (TOKEN_OVERLAP * VAE_RATIO_T > FRAME_PRE_PADDING)
    TOKEN_OVERLAP * VAE_RATIO_T - FRAME_PRE_PADDING
else
    0;

pub const IMAGENET_MEAN = [3]f32{ 0.485, 0.456, 0.406 };
pub const IMAGENET_STD = [3]f32{ 0.229, 0.224, 0.225 };

/// Decoder geometry (ViT3DDecoder defaults, confirmed against the checkpoint).
pub const DecCfg = struct {
    layers: u32 = 36,
    heads: u32 = 32,
    head_dim: u32 = 64,
    in_channels: u32 = 24,
    out_channels: u32 = 3,
    patch_size: u32 = VAE_RATIO,
    patch_size_t: u32 = VAE_RATIO_T,
    rope_theta: f64 = 100.0,
    rope_dim_ratio: f64 = 0.75,
    num_register_tokens: u32 = 4,
    eps: f32 = 1e-5,

    pub fn dim(self: DecCfg) u32 {
        return self.heads * self.head_dim;
    }
    /// RotaryEmbeddingND(dim_head * rope_dim_ratio) with n_dim=3: the inv_freq
    /// arange step is 2*n_dim/dim, so the count is ceil(dim / (2*n_dim)).
    pub fn ropeFreqs(self: DecCfg) u32 {
        const rd: u32 = @intFromFloat(@as(f64, @floatFromInt(self.head_dim)) * self.rope_dim_ratio);
        return (rd + 5) / 6;
    }
    /// Rotated width per head: 3 axes x freqs x 2 (split-half pairing).
    pub fn rotDim(self: DecCfg) u32 {
        return self.ropeFreqs() * 3 * 2;
    }
    pub fn outPatchDim(self: DecCfg) u32 {
        return self.out_channels * self.patch_size_t * self.patch_size * self.patch_size;
    }
};

// ── Temporal decode plan (pure arithmetic, hermetically tested) ─────────────

pub const TemporalPlan = struct {
    /// Latent tokens appended by repeating the last one.
    pad_tokens: u32,
    /// Chunks the decoder is invoked for.
    num_chunks: u32,
    /// Frames the assembled output holds after trimming the pad's contribution.
    output_frames: u32,
    /// Latent length after padding.
    padded_len: u32,
};

/// Mirrors `decode_temporal`'s head plus `_decode_temporal_frame_plan`.
pub fn planTemporal(latent_t: u32) TemporalPlan {
    var pseudo = latent_t + TOKEN_DROP;
    var pad: u32 = 0;
    const rem = pseudo % TOKENS_CHUNK_SIZE;
    if (rem != 0) {
        pad = TOKENS_CHUNK_SIZE - rem;
        pseudo += pad;
    }
    var num_chunks = pseudo / TOKENS_CHUNK_SIZE - @as(u32, @intFromBool(TOKEN_DROP > 0));
    if (num_chunks < 1) {
        // Too few tokens for one chunk (latent_t == 2): pad a whole extra chunk.
        pad += TOKENS_CHUNK_SIZE;
        num_chunks += 1;
    }
    const padded_len = latent_t + pad;
    return .{
        .pad_tokens = pad,
        .num_chunks = num_chunks,
        .output_frames = framePlan(padded_len, num_chunks, pad),
        .padded_len = padded_len,
    };
}

fn framePlan(z_len: u32, num_chunks: u32, pad_tokens: u32) u32 {
    const chunk_dec = TOKENS_CHUNK_SIZE * VAE_RATIO_T;
    const split_count: u32 = @as(u32, @intFromBool(TOKEN_DROP > 0)) + 1;
    var total: u32 = 0;
    var final_overlap: u32 = 0;
    for (0..num_chunks) |i| {
        const t_start = @as(u32, @intCast(i)) * TOKENS_CHUNK_SIZE;
        const t_end = t_start + TOKENS_CHUNK_SIZE + TOKEN_OVERLAP;
        const clip_tokens = @min(t_end, z_len) -| @min(t_start, z_len);
        const clip_frames = clip_tokens * VAE_RATIO_T;
        for (0..split_count) |j| {
            const f_start = @as(u32, @intCast(j)) * chunk_dec;
            const f_end = @min(f_start + chunk_dec, clip_frames);
            const frames = (f_end -| f_start) -| FRAME_PRE_PADDING;
            if (j == 0) total += frames else final_overlap = frames;
        }
    }
    total += final_overlap;
    return total - padFrames(z_len, pad_tokens);
}

fn padFrames(z_len: u32, pad_tokens: u32) u32 {
    if (pad_tokens == 0) return 0;
    const intra_tail = CLIP_LENGTH % VAE_RATIO_T;
    if (intra_tail == 0) return pad_tokens * VAE_RATIO_T;
    const before = z_len - pad_tokens;
    var sum: u32 = 0;
    for (0..pad_tokens) |k| {
        sum += if ((before + @as(u32, @intCast(k))) % TOKENS_CHUNK_SIZE == 0) intra_tail else VAE_RATIO_T;
    }
    return sum;
}

/// Whether a pixel extent decodes as ONE spatial tile. Above this the
/// reference's tiled path renormalizes each tile's rope coordinates, which we
/// do not implement — see the file header.
pub fn fitsSingleTile(pixel_extent: u32) bool {
    return TILE_SIZE >= pixel_extent;
}

// ── mlx helpers ─────────────────────────────────────────────────────────────

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
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
fn splitEqual(x: mlx.mlx_array, n: usize, axis: c_int, out: []mlx.mlx_array, s: S) !void {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_split(&vec, x, @intCast(n), axis, s));
    for (0..n) |i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, vec, i));
        out[i] = o;
    }
}
fn sliceAxis(x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    var start: [8]c_int = undefined;
    var stop: [8]c_int = undefined;
    var step: [8]c_int = undefined;
    const nd = shp.len;
    for (0..nd) |i| {
        start[i] = 0;
        stop[i] = @intCast(shp[i]);
        step[i] = 1;
    }
    start[axis] = lo;
    stop[axis] = hi;
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &start, nd, &stop, nd, &step, nd, s));
    return contig(o, s);
}
fn rmsNormLast(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
fn layerNormLast(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, eps, s));
    return o;
}
fn siluA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    defer _ = mlx.mlx_array_free(o);
    return mulA(x, o, s);
}
fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[minimax-h3-vae] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingMiniMaxH3VaeWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn ownAs(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, suffix: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix });
    defer a.free(key);
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, dt, s);
}
/// Linear stored [out, in]: pre-transposed at load so the hot path is a matmul.
fn loadLinT(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const raw = try ownAs(w, a, prefix, ".weight", dt, s);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    return contig(t, s);
}
fn linT(x: mlx.mlx_array, wt: mlx.mlx_array, bias: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
    return o;
}

// ── Decoder weights ─────────────────────────────────────────────────────────

const DecBlockW = struct {
    norm1: mlx.mlx_array,
    norm2: mlx.mlx_array,
    scale1: mlx.mlx_array,
    scale2: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_b: mlx.mlx_array,
    w1: mlx.mlx_array,
    w1_b: mlx.mlx_array,
    w2: mlx.mlx_array,
    w2_b: mlx.mlx_array,

    fn deinit(self: *DecBlockW) void {
        inline for (.{ self.norm1, self.norm2, self.scale1, self.scale2, self.qkv_w, self.qkv_b, self.out_w, self.out_b, self.w1, self.w1_b, self.w2, self.w2_b }) |f|
            _ = mlx.mlx_array_free(f);
    }
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    cfg: DecCfg,
    dtype: mlx.mlx_dtype,
    x_embed_w: mlx.mlx_array,
    x_embed_b: mlx.mlx_array,
    register_tokens: mlx.mlx_array,
    blocks: []DecBlockW,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    proj_out_w: mlx.mlx_array,
    proj_out_b: mlx.mlx_array,
    pq_w: mlx.mlx_array, // post_quant_conv, [24,24,1,1,1] -> used as [24,24]
    pq_b: mlx.mlx_array,
    latents_mean: mlx.mlx_array,
    latents_std: mlx.mlx_array,

    pub fn load(allocator: std.mem.Allocator, w: *const Weights, cfg: DecCfg, dt: mlx.mlx_dtype, s: S) !Decoder {
        var self: Decoder = undefined;
        self.allocator = allocator;
        self.cfg = cfg;
        self.dtype = dt;
        const a = allocator;

        self.x_embed_w = try loadLinT(w, a, "decoder.x_embedder", dt, s);
        self.x_embed_b = try ownAs(w, a, "decoder.x_embedder", ".bias", dt, s);
        self.register_tokens = try ownAs(w, a, "decoder", ".register_tokens", dt, s);
        self.norm_out_w = try ownAs(w, a, "decoder.norm_out", ".weight", dt, s);
        self.norm_out_b = try ownAs(w, a, "decoder.norm_out", ".bias", dt, s);
        self.proj_out_w = try loadLinT(w, a, "decoder.proj_out", dt, s);
        self.proj_out_b = try ownAs(w, a, "decoder.proj_out", ".bias", dt, s);

        // post_quant_conv is [24,24,1,1,1]: a 1x1x1 conv is a per-channel mix,
        // so it collapses to a [24,24] matmul on the channel axis.
        const pq_raw = try ownAs(w, a, "post_quant_conv", ".weight", dt, s);
        defer _ = mlx.mlx_array_free(pq_raw);
        const pq2 = try reshape(pq_raw, &[_]c_int{ @intCast(cfg.in_channels), @intCast(cfg.in_channels) }, s);
        defer _ = mlx.mlx_array_free(pq2);
        const pqt = try transpose(pq2, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(pqt);
        self.pq_w = try contig(pqt, s);
        self.pq_b = try ownAs(w, a, "post_quant_conv", ".bias", dt, s);

        // Kept f32: they scale the latent before anything else runs, and the
        // checkpoint stores them as plain per-channel tables.
        self.latents_mean = try ownAs(w, a, "latents_mean", "", mlx.mlx_dtype.float32, s);
        self.latents_std = try ownAs(w, a, "latents_std", "", mlx.mlx_dtype.float32, s);

        self.blocks = try a.alloc(DecBlockW, cfg.layers);
        for (self.blocks, 0..) |*b, i| {
            const p = try std.fmt.allocPrint(a, "decoder.transformer_blocks.{d}", .{i});
            defer a.free(p);
            const attn_p = try std.fmt.allocPrint(a, "{s}.attn.to_qkv", .{p});
            defer a.free(attn_p);
            const out_p = try std.fmt.allocPrint(a, "{s}.attn.to_out", .{p});
            defer a.free(out_p);
            const w1_p = try std.fmt.allocPrint(a, "{s}.ff.w1", .{p});
            defer a.free(w1_p);
            const w2_p = try std.fmt.allocPrint(a, "{s}.ff.w2", .{p});
            defer a.free(w2_p);
            b.* = .{
                .norm1 = try ownAs(w, a, p, ".norm1.weight", dt, s),
                .norm2 = try ownAs(w, a, p, ".norm2.weight", dt, s),
                .scale1 = try ownAs(w, a, p, ".scale1", dt, s),
                .scale2 = try ownAs(w, a, p, ".scale2", dt, s),
                .qkv_w = try loadLinT(w, a, attn_p, dt, s),
                .qkv_b = try ownAs(w, a, attn_p, ".bias", dt, s),
                .out_w = try loadLinT(w, a, out_p, dt, s),
                .out_b = try ownAs(w, a, out_p, ".bias", dt, s),
                .w1 = try loadLinT(w, a, w1_p, dt, s),
                .w1_b = try ownAs(w, a, w1_p, ".bias", dt, s),
                .w2 = try loadLinT(w, a, w2_p, dt, s),
                .w2_b = try ownAs(w, a, w2_p, ".bias", dt, s),
            };
        }
        return self;
    }

    pub fn deinit(self: *Decoder) void {
        inline for (.{ self.x_embed_w, self.x_embed_b, self.register_tokens, self.norm_out_w, self.norm_out_b, self.proj_out_w, self.proj_out_b, self.pq_w, self.pq_b, self.latents_mean, self.latents_std }) |f|
            _ = mlx.mlx_array_free(f);
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
    }

    /// Normalized coordinates for one (T,H,W) extent, plus zeros for the
    /// suffix tokens. Each axis is `(arange(0.5, n)/n)*2 - 1`, so the values
    /// depend on the EXTENT — which is why a spatial tile is not the same as
    /// the corresponding slice of an untiled pass.
    fn tokenIds(self: *const Decoder, t: u32, h: u32, w_: u32, n_suffix: u32) ![]f32 {
        const n = @as(usize, t) * h * w_;
        const out = try self.allocator.alloc(f32, (n + n_suffix) * 3);
        errdefer self.allocator.free(out);
        @memset(out, 0);
        var i: usize = 0;
        for (0..t) |ti| {
            const tv: f32 = @floatCast((@as(f64, @floatFromInt(ti)) + 0.5) / @as(f64, @floatFromInt(t)) * 2.0 - 1.0);
            for (0..h) |hi| {
                const hv: f32 = @floatCast((@as(f64, @floatFromInt(hi)) + 0.5) / @as(f64, @floatFromInt(h)) * 2.0 - 1.0);
                for (0..w_) |wi| {
                    const wv: f32 = @floatCast((@as(f64, @floatFromInt(wi)) + 0.5) / @as(f64, @floatFromInt(w_)) * 2.0 - 1.0);
                    out[i * 3 + 0] = tv;
                    out[i * 3 + 1] = hv;
                    out[i * 3 + 2] = wv;
                    i += 1;
                }
            }
        }
        return out;
    }

    fn buildRope(self: *const Decoder, ids: []const f32, n_rows: usize, s: S) !h3.RopeTables {
        const nf = self.cfg.ropeFreqs();
        const half: usize = @as(usize, nf) * 3;
        const ang = try self.allocator.alloc(f32, n_rows * half);
        defer self.allocator.free(ang);
        const two_pi: f64 = 2.0 * std.math.pi;
        for (0..n_rows) |r| {
            for (0..3) |ax| {
                const p: f64 = ids[r * 3 + ax];
                for (0..nf) |j| {
                    // inv_freq = 1 / theta^(j * 2*n_dim/dim); with dim = 48 and
                    // n_dim = 3 the exponent step is 0.125.
                    const step = 2.0 * 3.0 / (@as(f64, @floatFromInt(self.cfg.head_dim)) * self.cfg.rope_dim_ratio);
                    const inv = 1.0 / std.math.pow(f64, self.cfg.rope_theta, @as(f64, @floatFromInt(j)) * step);
                    ang[r * half + ax * nf + j] = @floatCast(two_pi * p * inv);
                }
            }
        }
        const shape = [_]c_int{ @intCast(n_rows), @intCast(half) };
        const arr = mlx.mlx_array_new_data(ang.ptr, &shape, 2, mlx.mlx_dtype.float32);
        defer _ = mlx.mlx_array_free(arr);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_cos(&c, arr, s));
        var sn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sn);
        try mlx.check(mlx.mlx_sin(&sn, arr, s));
        const bshape = [_]c_int{ 1, @intCast(n_rows), 1, @intCast(half) };
        const cb = try reshape(c, &bshape, s);
        defer _ = mlx.mlx_array_free(cb);
        const sb = try reshape(sn, &bshape, s);
        defer _ = mlx.mlx_array_free(sb);
        return .{ .cos = try astype(cb, self.dtype, s), .sin = try astype(sb, self.dtype, s) };
    }

    /// One decoder block: x += attn(rms(x)) * scale1; x += ff(rms(x)) * scale2.
    fn blockForward(self: *const Decoder, b: *const DecBlockW, x: mlx.mlx_array, rope: h3.RopeTables, s: S) !mlx.mlx_array {
        const cfg = self.cfg;
        const n: c_int = @intCast(mlx.getShape(x)[1]);
        const heads: c_int = @intCast(cfg.heads);
        const hd: c_int = @intCast(cfg.head_dim);

        const n1 = try rmsNormLast(x, b.norm1, cfg.eps, s);
        defer _ = mlx.mlx_array_free(n1);
        const qkv = try linT(n1, b.qkv_w, b.qkv_b, s);
        defer _ = mlx.mlx_array_free(qkv);
        // The reference views [B,S,3*inner] as [B,S,heads,3*head_dim] and then
        // chunks the LAST axis — so q/k/v are interleaved PER HEAD, not three
        // contiguous blocks. This differs from the DiT's split and is exactly
        // the kind of layout slip that still produces plausible output.
        const v4 = try reshape(qkv, &[_]c_int{ 1, n, heads, 3 * hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        var parts: [3]mlx.mlx_array = undefined;
        try splitEqual(v4, 3, 3, &parts, s);
        defer for (&parts) |*p| {
            _ = mlx.mlx_array_free(p.*);
        };

        var t: [3]mlx.mlx_array = undefined;
        var built: usize = 0;
        errdefer for (t[0..built]) |p| {
            _ = mlx.mlx_array_free(p);
        };
        for (0..3) |i| {
            var cur = try contig(parts[i], s);
            if (i < 2) {
                // norm_q / norm_k are elementwise_affine=False: a pure RMS
                // normalize with NO weight.
                const nn = try rmsNormLast(cur, mlx.mlx_array{ .ctx = null }, cfg.eps, s);
                _ = mlx.mlx_array_free(cur);
                cur = nn;
                const rp = try h3.applyRopePub(cur, rope, @intCast(cfg.rotDim() / 2), hd, s);
                _ = mlx.mlx_array_free(cur);
                cur = rp;
            }
            const tr = try transpose(cur, &[_]c_int{ 0, 2, 1, 3 }, s);
            _ = mlx.mlx_array_free(cur);
            defer _ = mlx.mlx_array_free(tr);
            t[i] = try contig(tr, s);
            built += 1;
        }
        defer for (&t) |*p| {
            _ = mlx.mlx_array_free(p.*);
        };

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, t[0], t[1], t[2], scale, "", null_a, null_a, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, n, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const ao = try linT(af, b.out_w, b.out_b, s);
        defer _ = mlx.mlx_array_free(ao);
        const gated = try mulA(ao, b.scale1, s);
        defer _ = mlx.mlx_array_free(gated);
        const h1 = try addA(x, gated, s);
        errdefer _ = mlx.mlx_array_free(h1);

        const n2 = try rmsNormLast(h1, b.norm2, cfg.eps, s);
        defer _ = mlx.mlx_array_free(n2);
        const y = try linT(n2, b.w1, b.w1_b, s);
        defer _ = mlx.mlx_array_free(y);
        var halves: [2]mlx.mlx_array = undefined;
        try splitEqual(y, 2, 2, &halves, s);
        defer for (&halves) |*p| {
            _ = mlx.mlx_array_free(p.*);
        };
        const g = try siluA(halves[0], s);
        defer _ = mlx.mlx_array_free(g);
        const act = try mulA(g, halves[1], s);
        defer _ = mlx.mlx_array_free(act);
        const ff = try linT(act, b.w2, b.w2_b, s);
        defer _ = mlx.mlx_array_free(ff);
        const gated2 = try mulA(ff, b.scale2, s);
        defer _ = mlx.mlx_array_free(gated2);
        const out = try addA(h1, gated2, s);
        _ = mlx.mlx_array_free(h1);
        return out;
    }

    /// One untiled ViT pass: latent [1, C, t, h, w] -> pixels [1, 3, t*4, h*16, w*16].
    pub fn decodePixels(self: *const Decoder, z: mlx.mlx_array, s: S) !mlx.mlx_array {
        const cfg = self.cfg;
        const shp = mlx.getShape(z);
        const t: u32 = @intCast(shp[2]);
        const hh: u32 = @intCast(shp[3]);
        const ww: u32 = @intCast(shp[4]);
        const n_patches: c_int = @intCast(t * hh * ww);
        const n_suffix: u32 = 1 + cfg.num_register_tokens;

        // [1,C,t,h,w] -> [1, t*h*w, C]
        const flat = try reshape(z, &[_]c_int{ 1, @intCast(cfg.in_channels), n_patches }, s);
        defer _ = mlx.mlx_array_free(flat);
        const tr = try transpose(flat, &[_]c_int{ 0, 2, 1 }, s);
        defer _ = mlx.mlx_array_free(tr);
        const trc = try contig(tr, s);
        defer _ = mlx.mlx_array_free(trc);
        const emb_in = try astype(trc, self.dtype, s);
        defer _ = mlx.mlx_array_free(emb_in);
        const h0 = try linT(emb_in, self.x_embed_w, self.x_embed_b, s);
        defer _ = mlx.mlx_array_free(h0);

        // Append the learned register tokens and ONE zero token; both carry
        // all-zero position ids.
        var zero_shape = [_]c_int{ 1, 1, @intCast(cfg.dim()) };
        var zeros = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zeros);
        try mlx.check(mlx.mlx_zeros(&zeros, &zero_shape, 3, self.dtype, s));
        var h = try concat(&[_]mlx.mlx_array{ h0, self.register_tokens, zeros }, 1, s);
        errdefer _ = mlx.mlx_array_free(h);

        const n_rows: usize = @as(usize, @intCast(n_patches)) + n_suffix;
        const ids = try self.tokenIds(t, hh, ww, n_suffix);
        defer self.allocator.free(ids);
        var rope = try self.buildRope(ids, n_rows, s);
        defer rope.deinit();

        for (self.blocks) |*b| {
            const nh = try self.blockForward(b, h, rope, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }

        const no = try layerNormLast(h, self.norm_out_w, self.norm_out_b, cfg.eps, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(no);
        const po = try linT(no, self.proj_out_w, self.proj_out_b, s);
        defer _ = mlx.mlx_array_free(po);
        // Drop the suffix rows before unpatchifying.
        const kept = try sliceAxis(po, 1, 0, n_patches, s);
        defer _ = mlx.mlx_array_free(kept);

        // [1, t*h*w, C*pt*ph*pw] -> [1, C, t*pt, h*ph, w*pw]
        const pt: c_int = @intCast(cfg.patch_size_t);
        const ps: c_int = @intCast(cfg.patch_size);
        const oc: c_int = @intCast(cfg.out_channels);
        const v = try reshape(kept, &[_]c_int{ 1, @intCast(t), @intCast(hh), @intCast(ww), oc, pt, ps, ps }, s);
        defer _ = mlx.mlx_array_free(v);
        const perm = try transpose(v, &[_]c_int{ 0, 4, 1, 5, 2, 6, 3, 7 }, s);
        defer _ = mlx.mlx_array_free(perm);
        const permc = try contig(perm, s);
        defer _ = mlx.mlx_array_free(permc);
        return reshape(permc, &[_]c_int{ 1, oc, @intCast(t * cfg.patch_size_t), @intCast(hh * cfg.patch_size), @intCast(ww * cfg.patch_size) }, s);
    }
};

/// Cross-fade `a`'s tail into `b`'s head over `extent` frames on axis 2.
/// Linear ramp, matching `blend`.
fn blendFrames(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, extent: u32, alloc: std.mem.Allocator, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const an: u32 = @intCast(mlx.getShape(a_arr)[2]);
    const bn: u32 = @intCast(mlx.getShape(b_arr)[2]);
    const e = @min(@min(an, bn), extent);
    if (e == 0) return contig(b_arr, s);

    const wbuf = try alloc.alloc(f32, e);
    defer alloc.free(wbuf);
    for (0..e) |i| wbuf[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(e));
    const wshape = [_]c_int{ 1, 1, @intCast(e), 1, 1 };
    const wb_arr = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(wb_arr);
    const wb = try astype(wb_arr, dt, s);
    defer _ = mlx.mlx_array_free(wb);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var wa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wa);
    try mlx.check(mlx.mlx_subtract(&wa, one, wb, s));

    const a_tail = try sliceAxis(a_arr, 2, @intCast(an - e), @intCast(an), s);
    defer _ = mlx.mlx_array_free(a_tail);
    const b_head = try sliceAxis(b_arr, 2, 0, @intCast(e), s);
    defer _ = mlx.mlx_array_free(b_head);
    const ta = try mulA(a_tail, wa, s);
    defer _ = mlx.mlx_array_free(ta);
    const tb = try mulA(b_head, wb, s);
    defer _ = mlx.mlx_array_free(tb);
    const mixed = try addA(ta, tb, s);
    defer _ = mlx.mlx_array_free(mixed);
    if (bn == e) return contig(mixed, s);
    const rest = try sliceAxis(b_arr, 2, @intCast(e), @intCast(bn), s);
    defer _ = mlx.mlx_array_free(rest);
    return concat(&[_]mlx.mlx_array{ mixed, rest }, 2, s);
}

/// Full decode: normalized latents [1,24,T,H,W] -> pixels [1,3,frames,H*16,W*16]
/// in [-1, 1].
///
/// REFUSES a canvas that would need spatial tiling rather than silently
/// decoding it untiled — the reference renormalizes each tile's rope
/// coordinates, so an untiled pass at 864 wide is a different computation, not
/// an approximation of the same one.
pub fn decode(dec: *const Decoder, z_norm: mlx.mlx_array, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(z_norm);
    const latent_t: u32 = @intCast(shp[2]);
    const lat_h: u32 = @intCast(shp[3]);
    const lat_w: u32 = @intCast(shp[4]);
    if (!fitsSingleTile(lat_h * VAE_RATIO) or !fitsSingleTile(lat_w * VAE_RATIO)) {
        log.err("[minimax-h3-vae] {d}x{d} px needs spatial tiling (tile {d}); not implemented\n", .{ lat_w * VAE_RATIO, lat_h * VAE_RATIO, TILE_SIZE });
        return error.TilingUnsupported;
    }

    // Denormalize: z * std + mean, then the 1x1x1 post-quant mix.
    const zf = try astype(z_norm, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(zf);
    const cshape = [_]c_int{ 1, @intCast(dec.cfg.in_channels), 1, 1, 1 };
    const lm = try reshape(dec.latents_mean, &cshape, s);
    defer _ = mlx.mlx_array_free(lm);
    const ls = try reshape(dec.latents_std, &cshape, s);
    defer _ = mlx.mlx_array_free(ls);
    const scaled = try mulA(zf, ls, s);
    defer _ = mlx.mlx_array_free(scaled);
    const zden = try addA(scaled, lm, s);
    defer _ = mlx.mlx_array_free(zden);

    // post_quant_conv on the channel axis: [1,C,T,H,W] -> [1,N,C] -> mix -> back
    const n_all: c_int = @intCast(latent_t * lat_h * lat_w);
    const zflat = try reshape(zden, &[_]c_int{ 1, @intCast(dec.cfg.in_channels), n_all }, s);
    defer _ = mlx.mlx_array_free(zflat);
    const ztr = try transpose(zflat, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(ztr);
    const ztrc = try contig(ztr, s);
    defer _ = mlx.mlx_array_free(ztrc);
    const ztd = try astype(ztrc, dec.dtype, s);
    defer _ = mlx.mlx_array_free(ztd);
    const mixed = try linT(ztd, dec.pq_w, dec.pq_b, s);
    defer _ = mlx.mlx_array_free(mixed);
    const back = try transpose(mixed, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(back);
    const backc = try contig(back, s);
    defer _ = mlx.mlx_array_free(backc);
    var z = try reshape(backc, &[_]c_int{ 1, @intCast(dec.cfg.in_channels), @intCast(latent_t), @intCast(lat_h), @intCast(lat_w) }, s);
    defer _ = mlx.mlx_array_free(z);

    const plan = planTemporal(latent_t);
    // Pad by repeating the LAST latent token.
    if (plan.pad_tokens > 0) {
        const last = try sliceAxis(z, 2, @intCast(latent_t - 1), @intCast(latent_t), s);
        defer _ = mlx.mlx_array_free(last);
        var pieces = try alloc.alloc(mlx.mlx_array, 1 + plan.pad_tokens);
        defer alloc.free(pieces);
        pieces[0] = z;
        for (1..pieces.len) |i| pieces[i] = last;
        const padded = try concat(pieces, 2, s);
        _ = mlx.mlx_array_free(z);
        z = padded;
    }

    const chunk_dec = TOKENS_CHUNK_SIZE * VAE_RATIO_T;
    const split_count: u32 = @as(u32, @intFromBool(TOKEN_DROP > 0)) + 1;
    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |p| _ = mlx.mlx_array_free(p);
        parts.deinit(alloc);
    }
    var overlap: ?mlx.mlx_array = null;
    errdefer if (overlap) |o| {
        _ = mlx.mlx_array_free(o);
    };

    const z_len = plan.padded_len;
    for (0..plan.num_chunks) |i| {
        const t_start = @as(u32, @intCast(i)) * TOKENS_CHUNK_SIZE;
        const t_end = @min(t_start + TOKENS_CHUNK_SIZE + TOKEN_OVERLAP, z_len);
        if (t_start >= t_end) continue;
        const clip_z = try sliceAxis(z, 2, @intCast(t_start), @intCast(t_end), s);
        defer _ = mlx.mlx_array_free(clip_z);
        const clip_dec = try dec.decodePixels(clip_z, s);
        defer _ = mlx.mlx_array_free(clip_dec);
        const clip_frames: u32 = @intCast(mlx.getShape(clip_dec)[2]);

        for (0..split_count) |j| {
            const f_start = @as(u32, @intCast(j)) * chunk_dec;
            if (f_start >= clip_frames) continue;
            const f_end = @min(f_start + chunk_dec, clip_frames);
            if (f_end - f_start <= FRAME_PRE_PADDING) continue;
            // Every decoded chunk drops `frame_pre_padding` frames off the
            // FRONT — the VAE's clip length is not a multiple of its temporal
            // ratio, so those frames are the previous clip's tail.
            const piece = try sliceAxis(clip_dec, 2, @intCast(f_start + FRAME_PRE_PADDING), @intCast(f_end), s);
            if (j == 0) {
                if (overlap) |o| {
                    defer _ = mlx.mlx_array_free(o);
                    defer _ = mlx.mlx_array_free(piece);
                    overlap = null;
                    const blended = try blendFrames(o, piece, FRAME_OVERLAP, alloc, dec.dtype, s);
                    try parts.append(alloc, blended);
                } else {
                    try parts.append(alloc, piece);
                }
            } else {
                if (overlap) |o| _ = mlx.mlx_array_free(o);
                overlap = piece;
            }
        }
        if (i == plan.num_chunks - 1) {
            if (overlap) |o| {
                try parts.append(alloc, o);
                overlap = null;
            }
        }
    }
    if (parts.items.len == 0) return error.EmptyDecode;

    var dec_all = try concat(parts.items, 2, s);
    errdefer _ = mlx.mlx_array_free(dec_all);
    const have: u32 = @intCast(mlx.getShape(dec_all)[2]);
    if (have > plan.output_frames) {
        const trimmed = try sliceAxis(dec_all, 2, 0, @intCast(plan.output_frames), s);
        _ = mlx.mlx_array_free(dec_all);
        dec_all = trimmed;
    }

    // Pixel denormalization: ImageNet stats, clamp to [0,1], then to [-1,1].
    const df = try astype(dec_all, mlx.mlx_dtype.float32, s);
    _ = mlx.mlx_array_free(dec_all);
    defer _ = mlx.mlx_array_free(df);
    const pshape = [_]c_int{ 1, 3, 1, 1, 1 };
    const pm_arr = mlx.mlx_array_new_data(&IMAGENET_MEAN, &pshape, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(pm_arr);
    const ps_arr = mlx.mlx_array_new_data(&IMAGENET_STD, &pshape, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(ps_arr);
    const m1 = try mulA(df, ps_arr, s);
    defer _ = mlx.mlx_array_free(m1);
    const a1 = try addA(m1, pm_arr, s);
    defer _ = mlx.mlx_array_free(a1);
    const lo = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var cl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cl);
    try mlx.check(mlx.mlx_clip(&cl, a1, lo, hi, s));
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    const scaled2 = try mulA(cl, two, s);
    defer _ = mlx.mlx_array_free(scaled2);
    const one2 = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one2);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&out, scaled2, one2, s));
    return out;
}

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

test "minimax h3 vae: temporal constants follow from the clip length" {
    // These are derived, not typed in: a future clip_length change must move
    // them together or the chunk walk desynchronizes from the blend.
    try testing.expectEqual(@as(u32, 3), FRAME_PRE_PADDING);
    try testing.expectEqual(@as(u32, 5), TOKENS_CHUNK_SIZE);
    try testing.expectEqual(@as(u32, 2), TOKEN_OVERLAP);
    try testing.expectEqual(@as(u32, 5), FRAME_OVERLAP);
}

test "minimax h3 vae: temporal plan reproduces the frame count" {
    // The plan must return exactly the frame count the DiT was asked for:
    // latent_t comes from videoLatentT, so decode has to invert it.
    const cases = [_][2]u32{
        // frame_count, latent_t
        .{ 5, 2 },
        .{ 22, 7 },
        .{ 56, 17 },
        .{ 124, 37 },
        .{ 362, 107 },
    };
    for (cases) |c| {
        const frame_count = c[0];
        const latent_t = c[1];
        try testing.expectEqual(latent_t, h3.videoLatentT(frame_count));
        const plan = planTemporal(latent_t);
        try testing.expectEqual(frame_count, plan.output_frames);
        try testing.expect(plan.num_chunks >= 1);
        try testing.expectEqual(latent_t + plan.pad_tokens, plan.padded_len);
    }
}

test "minimax h3 vae: the shortest clip still forms one chunk" {
    // latent_t == 2 is below one chunk; the reference pads a whole extra chunk
    // rather than emitting zero chunks (which would decode nothing at all).
    const plan = planTemporal(2);
    try testing.expectEqual(@as(u32, 1), plan.num_chunks);
    try testing.expect(plan.pad_tokens >= TOKENS_CHUNK_SIZE);
    try testing.expectEqual(@as(u32, 5), plan.output_frames);
}

test "minimax h3 vae: single-tile gate matches the reference's split" {
    // 256 is the tile extent, and split_tiles returns one tile when
    // tile_size >= input_len — so 256 itself is single-tile, 257 is not.
    try testing.expect(fitsSingleTile(256));
    try testing.expect(fitsSingleTile(128));
    try testing.expect(!fitsSingleTile(257));
    // A 256x256 canvas decodes untiled; the 768p native canvas does not, which
    // is exactly why the first bring-up target is 256x256.
    try testing.expect(fitsSingleTile(256) and !fitsSingleTile(864));
}

test "minimax h3 vae: decoder geometry matches the checkpoint" {
    const cfg = DecCfg{};
    try testing.expectEqual(@as(u32, 2048), cfg.dim());
    // rope_dim_ratio 0.75 of head_dim 64 = 48; arange(0, 1, 2*3/48) = 8 freqs.
    try testing.expectEqual(@as(u32, 8), cfg.ropeFreqs());
    // 8 freqs x 3 axes x 2 = 48 of head_dim 64 rotated, 16 pass through.
    try testing.expectEqual(@as(u32, 48), cfg.rotDim());
    try testing.expect(cfg.rotDim() < cfg.head_dim);
    // proj_out width: 3 channels x 4 frames x 16 x 16.
    try testing.expectEqual(@as(u32, 3072), cfg.outPatchDim());
}
