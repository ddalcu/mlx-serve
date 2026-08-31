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
//!     same region's positions in an untiled pass — decoding a 864-wide canvas
//!     in one go is a DIFFERENT computation, not an approximation of the tiled
//!     one. At or below 256 px `splitTiles` returns a single tile and the two
//!     agree; above it `decodeSpatial` walks the grid, cross-fades each tile
//!     into its up/left neighbours and trims so the kept extents tile the
//!     canvas exactly.
//!
//! Ported from ComfyUI `comfy/ldm/minimax/vae.py`.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const h3 = @import("minimax_h3.zig");
const preview = @import("preview.zig");

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

/// Whether a pixel extent decodes as ONE spatial tile.
pub fn fitsSingleTile(pixel_extent: u32) bool {
    return TILE_SIZE >= pixel_extent;
}

// ── Encoder geometry (EncoderFCN3D: ch=128, ch_mult (1,2,2,4,4,8)) ──────────

pub const ENC_LEVELS: u32 = 6;
pub const ENC_CH: u32 = 128;
pub const ENC_CH_MULT = [ENC_LEVELS]u32{ 1, 2, 2, 4, 4, 8 };
pub const ENC_SPACE_DOWN = [ENC_LEVELS]u32{ 2, 2, 2, 2, 1, 1 };
pub const ENC_TIME_DOWN = [ENC_LEVELS]u32{ 1, 2, 2, 1, 1, 1 };
pub const ENC_RES_BLOCKS: u32 = 2;

pub fn encBlockMid(level: usize) u32 {
    return ENC_CH * ENC_CH_MULT[level];
}
pub fn encBlockIn(level: usize) u32 {
    return if (level == 0) encBlockMid(0) else encBlockMid(level - 1);
}
/// A level carries a downsample conv iff space*time strides > 1.
pub fn encHasDown(level: usize) bool {
    return ENC_SPACE_DOWN[level] * ENC_TIME_DOWN[level] > 1;
}

/// Minimum tile overlap in pixels (`vae_tile_overlap_min`).
pub const TILE_OVERLAP_MIN: u32 = 64;

/// Spatial tile plan for one axis, in PIXELS. Mirrors `split_tiles`.
///
/// Tiles are all TILE_SIZE long; the overlaps absorb the difference, are grown
/// in whole `vae_ratio` units so every boundary lands on a latent row, and are
/// spread round-robin so no single seam carries the slack.
pub const TilePlan = struct {
    starts: []u32,
    overlaps: []u32,
    len: u32, // every tile is this long
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TilePlan) void {
        self.allocator.free(self.starts);
        self.allocator.free(self.overlaps);
    }
    pub fn count(self: *const TilePlan) usize {
        return self.starts.len;
    }
};

pub fn splitTiles(allocator: std.mem.Allocator, input_len: u32) !TilePlan {
    if (TILE_SIZE >= input_len) {
        const starts = try allocator.alloc(u32, 1);
        starts[0] = 0;
        return .{ .starts = starts, .overlaps = try allocator.alloc(u32, 0), .len = input_len, .allocator = allocator };
    }
    var n: u32 = (input_len + TILE_SIZE - 1) / TILE_SIZE;
    var remaining: i64 = 0;
    while (true) {
        remaining = @as(i64, TILE_SIZE) * n - @as(i64, TILE_OVERLAP_MIN) * (n - 1) - @as(i64, input_len);
        if (remaining < 0) n += 1 else break;
    }
    const overlaps = try allocator.alloc(u32, n - 1);
    errdefer allocator.free(overlaps);
    @memset(overlaps, TILE_OVERLAP_MIN);
    const units: u32 = @intCast(@divFloor(remaining, @as(i64, VAE_RATIO)));
    for (0..units) |i| overlaps[i % overlaps.len] += VAE_RATIO;

    const starts = try allocator.alloc(u32, n);
    errdefer allocator.free(starts);
    starts[0] = 0;
    for (0..n - 1) |i| starts[i + 1] = starts[i] + TILE_SIZE - overlaps[i];
    return .{ .starts = starts, .overlaps = overlaps, .len = TILE_SIZE, .allocator = allocator };
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
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, t[0], t[1], t[2], scale, "", null_a, null_a, false, s));
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
fn blendAxis(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, extent: u32, axis: usize, alloc: std.mem.Allocator, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const an: u32 = @intCast(mlx.getShape(a_arr)[axis]);
    const bn: u32 = @intCast(mlx.getShape(b_arr)[axis]);
    const e = @min(@min(an, bn), extent);
    if (e == 0) return contig(b_arr, s);

    const wbuf = try alloc.alloc(f32, e);
    defer alloc.free(wbuf);
    for (0..e) |i| wbuf[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(e));
    // Broadcast the ramp along `axis` only.
    var wshape = [_]c_int{ 1, 1, 1, 1, 1 };
    wshape[axis] = @intCast(e);
    const wb_arr = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(wb_arr);
    const wb = try astype(wb_arr, dt, s);
    defer _ = mlx.mlx_array_free(wb);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var wa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wa);
    try mlx.check(mlx.mlx_subtract(&wa, one, wb, s));

    const a_tail = try sliceAxis(a_arr, axis, @intCast(an - e), @intCast(an), s);
    defer _ = mlx.mlx_array_free(a_tail);
    const b_head = try sliceAxis(b_arr, axis, 0, @intCast(e), s);
    defer _ = mlx.mlx_array_free(b_head);
    const ta = try mulA(a_tail, wa, s);
    defer _ = mlx.mlx_array_free(ta);
    const tb = try mulA(b_head, wb, s);
    defer _ = mlx.mlx_array_free(tb);
    const mixed = try addA(ta, tb, s);
    defer _ = mlx.mlx_array_free(mixed);
    if (bn == e) return contig(mixed, s);
    const rest = try sliceAxis(b_arr, axis, @intCast(e), @intCast(bn), s);
    defer _ = mlx.mlx_array_free(rest);
    return concat(&[_]mlx.mlx_array{ mixed, rest }, @intCast(axis), s);
}

/// One latent chunk -> pixels, spatially TILED when the canvas exceeds the
/// 256-px tile extent.
///
/// Tiles are decoded independently (each renormalizes its own rope coordinates,
/// which is why this is not equivalent to one big pass), cross-faded into their
/// up/left neighbours over the planned overlap, then trimmed so the kept extents
/// tile the canvas exactly. Row-major assembly by concatenation rather than
/// writes into a mutable canvas.
fn decodeSpatial(dec: *const Decoder, z: mlx.mlx_array, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(z);
    const lat_h: u32 = @intCast(shp[3]);
    const lat_w: u32 = @intCast(shp[4]);
    const px_h = lat_h * VAE_RATIO;
    const px_w = lat_w * VAE_RATIO;
    if (fitsSingleTile(px_h) and fitsSingleTile(px_w)) return dec.decodePixels(z, s);

    var yp = try splitTiles(alloc, px_h);
    defer yp.deinit();
    var xp = try splitTiles(alloc, px_w);
    defer xp.deinit();

    // Bottom strips of the previous row, one per column, captured BEFORE that
    // row's tiles were blended (matching the reference's clone order).
    var row_tails = try alloc.alloc(?mlx.mlx_array, xp.count());
    defer alloc.free(row_tails);
    @memset(row_tails, null);
    defer for (row_tails) |t| if (t) |v| {
        _ = mlx.mlx_array_free(v);
    };

    var rows = try alloc.alloc(mlx.mlx_array, yp.count());
    defer alloc.free(rows);
    var rows_built: usize = 0;
    errdefer for (rows[0..rows_built]) |r| {
        _ = mlx.mlx_array_free(r);
    };

    for (yp.starts, 0..) |y0, i| {
        const zi = y0 / VAE_RATIO;
        const zl = yp.len / VAE_RATIO;
        var new_tails = try alloc.alloc(?mlx.mlx_array, xp.count());
        defer alloc.free(new_tails);
        @memset(new_tails, null);

        var cols = try alloc.alloc(mlx.mlx_array, xp.count());
        defer alloc.free(cols);
        var cols_built: usize = 0;
        errdefer for (cols[0..cols_built]) |c| {
            _ = mlx.mlx_array_free(c);
        };
        var left_tail: ?mlx.mlx_array = null;
        defer if (left_tail) |v| {
            _ = mlx.mlx_array_free(v);
        };

        for (xp.starts, 0..) |x0, j| {
            const zj = x0 / VAE_RATIO;
            const zw = xp.len / VAE_RATIO;
            const sub_h = try sliceAxis(z, 3, @intCast(zi), @intCast(zi + zl), s);
            defer _ = mlx.mlx_array_free(sub_h);
            const sub = try sliceAxis(sub_h, 4, @intCast(zj), @intCast(zj + zw), s);
            defer _ = mlx.mlx_array_free(sub);
            var tile = try dec.decodePixels(sub, s);
            errdefer _ = mlx.mlx_array_free(tile);

            // Capture the strips the NEXT row/column will blend against, taken
            // from the unblended tile.
            const th: u32 = @intCast(mlx.getShape(tile)[3]);
            const tw: u32 = @intCast(mlx.getShape(tile)[4]);
            if (i + 1 < yp.count()) {
                new_tails[j] = try sliceAxis(tile, 3, @intCast(th - yp.overlaps[i]), @intCast(th), s);
            }
            var next_left: ?mlx.mlx_array = null;
            if (j + 1 < xp.count()) {
                next_left = try sliceAxis(tile, 4, @intCast(tw - xp.overlaps[j]), @intCast(tw), s);
            }

            if (i > 0) {
                if (row_tails[j]) |prev| {
                    const b = try blendAxis(prev, tile, yp.overlaps[i - 1], 3, alloc, dec.dtype, s);
                    _ = mlx.mlx_array_free(tile);
                    tile = b;
                }
            }
            if (j > 0) {
                if (left_tail) |prev| {
                    const b = try blendAxis(prev, tile, xp.overlaps[j - 1], 4, alloc, dec.dtype, s);
                    _ = mlx.mlx_array_free(tile);
                    tile = b;
                }
            }
            if (left_tail) |v| _ = mlx.mlx_array_free(v);
            left_tail = next_left;

            // Trim the overlap that the next tile will own.
            if (i + 1 < yp.count()) {
                const h2: u32 = @intCast(mlx.getShape(tile)[3]);
                const t2 = try sliceAxis(tile, 3, 0, @intCast(h2 - yp.overlaps[i]), s);
                _ = mlx.mlx_array_free(tile);
                tile = t2;
            }
            if (j + 1 < xp.count()) {
                const w2: u32 = @intCast(mlx.getShape(tile)[4]);
                const t2 = try sliceAxis(tile, 4, 0, @intCast(w2 - xp.overlaps[j]), s);
                _ = mlx.mlx_array_free(tile);
                tile = t2;
            }
            cols[j] = tile;
            cols_built += 1;
        }
        rows[i] = try concat(cols, 4, s);
        rows_built += 1;
        for (cols) |c| _ = mlx.mlx_array_free(c);

        for (row_tails, 0..) |t, k| {
            if (t) |v| _ = mlx.mlx_array_free(v);
            row_tails[k] = new_tails[k];
            new_tails[k] = null;
        }
    }
    const outp = try concat(rows, 3, s);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    rows_built = 0;
    return outp;
}

/// Full decode: normalized latents [1,24,T,H,W] -> pixels [1,3,frames,H*16,W*16]
/// in [-1, 1].
pub fn decode(dec: *const Decoder, z_norm: mlx.mlx_array, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(z_norm);
    const latent_t: u32 = @intCast(shp[2]);
    const lat_h: u32 = @intCast(shp[3]);
    const lat_w: u32 = @intCast(shp[4]);

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
        const clip_dec = try decodeSpatial(dec, clip_z, alloc, s);
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
                    const blended = try blendAxis(o, piece, FRAME_OVERLAP, 2, alloc, dec.dtype, s);
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

test "minimax h3 vae: tile plans cover the canvas exactly" {
    const a = testing.allocator;
    // Every canvas we can generate must be covered with no gap and no double
    // count: sum(tile_len) - sum(overlaps) has to equal the extent, or the
    // assembled rows are the wrong width and the concat shape-errors.
    for ([_]u32{ 256, 288, 480, 512, 768, 864, 1024, 1344, 2048 }) |extent| {
        var plan = try splitTiles(a, extent);
        defer plan.deinit();
        try testing.expect(plan.count() >= 1);
        try testing.expectEqual(plan.count() - 1, plan.overlaps.len);

        // Sum the KEPT extent per tile, exactly as decodeSpatial trims it:
        // every tile but the last gives up its overlap to the next one.
        // Accumulating additively also means a wrong plan reports a wrong
        // number rather than wrapping a u32 subtraction into nonsense.
        var covered: u32 = 0;
        for (0..plan.count()) |j| {
            covered += if (j + 1 < plan.count()) plan.len - plan.overlaps[j] else plan.len;
        }
        testing.expectEqual(extent, covered) catch |e| {
            std.debug.print("extent {d}: {d} tiles of {d}, overlaps {any}, covered {d}\n", .{ extent, plan.count(), plan.len, plan.overlaps, covered });
            return e;
        };

        // Boundaries must land on LATENT rows, or the latent slice is not
        // expressible and the tile decodes a shifted region.
        for (plan.starts) |st| try testing.expectEqual(@as(u32, 0), st % VAE_RATIO);
        for (plan.overlaps) |o| {
            try testing.expectEqual(@as(u32, 0), o % VAE_RATIO);
            // A blend needs something to blend, and it must not consume a whole
            // tile.
            try testing.expect(o >= TILE_OVERLAP_MIN);
            try testing.expect(o < plan.len);
        }
        // Tiles advance strictly, and the last one ends exactly at the extent.
        for (1..plan.count()) |i| try testing.expect(plan.starts[i] > plan.starts[i - 1]);
        try testing.expectEqual(extent, plan.starts[plan.count() - 1] + plan.len);
    }
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

test "minimax h3 vae live: tiled decode seam energy stays at ambient level" {
    // The blend/trim path of `decodeSpatial` was only ever validated by "the
    // 480p clip looked right". This pins it numerically: a smooth synthetic
    // latent decoded through the TILED path (384px canvas -> 2x2 tiles, one
    // seam per axis at pixel 128) must show NO gradient spike at the seams —
    // broken blending (hard concat, off-by-one trim) reads as a step edge
    // there. Tiles are SEMANTIC (each renormalizes rope over its own extent),
    // so tile outputs genuinely differ and an unblended seam is visible.
    const raw_model = std.c.getenv("MINIMAX_H3_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    if (model_dir.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();

    const vae_path = try std.fmt.allocPrint(a, "{s}/video_vae.safetensors", .{model_dir});
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var dec = try Decoder.load(a, &vw, .{}, mlx.mlx_dtype.bfloat16, s);
    defer dec.deinit();
    _ = io;

    // Smooth low-frequency latent [1,24,2,24,24] -> 384x384, 5 frames. Channel
    // phases vary so every tile sees different content (a constant latent
    // makes all tiles identical and the seam check vacuous).
    const lat: u32 = 24;
    const lt: u32 = 2;
    const buf = try a.alloc(f32, 24 * lt * lat * lat);
    defer a.free(buf);
    for (0..24) |c| {
        const phase = @as(f64, @floatFromInt(c)) * 0.37;
        for (0..lt) |t| {
            for (0..lat) |y| {
                for (0..lat) |x| {
                    const fx = @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(lat));
                    const fy = @as(f64, @floatFromInt(y)) / @as(f64, @floatFromInt(lat));
                    const v = 0.6 * @sin(2.0 * std.math.pi * (fx + phase)) * @cos(2.0 * std.math.pi * (fy - phase));
                    buf[((c * lt + t) * lat + y) * lat + x] = @floatCast(v);
                }
            }
        }
    }
    const zshape = [_]c_int{ 1, 24, @intCast(lt), @intCast(lat), @intCast(lat) };
    const z = mlx.mlx_array_new_data(buf.ptr, &zshape, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(z);

    const pixels = try decode(&dec, z, a, s);
    defer _ = mlx.mlx_array_free(pixels);
    const pf = try astype(pixels, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(pf);
    try mlx.check(mlx.mlx_array_eval(pf));

    const shp = mlx.getShape(pf);
    const frames: usize = @intCast(shp[2]);
    const ph: usize = @intCast(shp[3]);
    const pw: usize = @intCast(shp[4]);
    try testing.expectEqual(@as(usize, 384), ph);
    try testing.expectEqual(@as(usize, 384), pw);
    const data = mlx.mlx_array_data_float32(pf) orelse return error.NoPixelData;
    const n_px = 3 * frames * ph * pw;

    // Column-gradient energy g[x] = mean |p[..,x+1] - p[..,x]|, and the row
    // analogue. splitTiles(384) = starts [0,128], overlap 128 -> the assembled
    // seam sits between columns 127|128 (gradient index 127).
    const gcol = try a.alloc(f64, pw - 1);
    defer a.free(gcol);
    @memset(gcol, 0);
    const grow = try a.alloc(f64, ph - 1);
    defer a.free(grow);
    @memset(grow, 0);
    var ci: usize = 0;
    while (ci < n_px) : (ci += ph * pw) {
        const plane = data[ci .. ci + ph * pw];
        for (0..ph) |y| {
            for (0..pw - 1) |x| gcol[x] += @abs(@as(f64, plane[y * pw + x + 1]) - @as(f64, plane[y * pw + x]));
        }
        for (0..ph - 1) |y| {
            for (0..pw) |x| grow[y] += @abs(@as(f64, plane[(y + 1) * pw + x]) - @as(f64, plane[y * pw + x]));
        }
    }

    const med = struct {
        fn f(alloc2: std.mem.Allocator, v: []const f64) !f64 {
            const c2 = try alloc2.dupe(f64, v);
            defer alloc2.free(c2);
            std.mem.sort(f64, c2, {}, std.sort.asc(f64));
            return c2[c2.len / 2];
        }
    }.f;
    const col_med = try med(a, gcol);
    const row_med = try med(a, grow);
    const seam = 127;
    std.debug.print("[h3-vae-seam] col seam={d:.5} med={d:.5}  row seam={d:.5} med={d:.5}\n", .{ gcol[seam], col_med, grow[seam], row_med });
    // A correct cross-fade leaves the seam column statistically ordinary; a
    // hard concat measured ~an order of magnitude over the ambient median.
    try testing.expect(gcol[seam] <= col_med * 3.0);
    try testing.expect(grow[seam] <= row_med * 3.0);
}

// ── Encoder (3D causal CNN) ─────────────────────────────────────────────────
//
// ONE path serves fl2va keyframes (T==1) and ref2va reference videos (T>1).
// The reference's CausalConv3d has a T==1 branch that truncates the temporal
// taps instead of convolving zero frames — but that is an OPTIMIZATION, not a
// different semantic: the causal front padding is all zeros, so the taps it
// skips contribute exactly 0.0. Running the zero-pad form at every T is
// therefore the same arithmetic with one implementation, which is worth more
// than the frames of multiply-by-zero it costs on a keyframe.
//
// Two things are semantic and must not be "optimized":
//   - temporal CHUNKING (17-frame clips, the length the VAE was trained on),
//     which is why a long video is not one pass, and
//   - spatial TILING, for the same reason on the decode side.
//
// Runs f32 end to end: the moments anchor the whole generation, so precision
// is the safe default here.

inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
inline fn divA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, a, b, s));
    return o;
}

/// ImageNet statistics — non-persistent buffers in the reference, so they are
/// NOT in the checkpoint and live here as the reference's own constants.
const PIXEL_MEAN = [3]f32{ 0.485, 0.456, 0.406 };
const PIXEL_STD = [3]f32{ 0.229, 0.224, 0.225 };

/// Reflect-pad H and W by 1 on [1, T, H, W, C] (torch reflect: mirror
/// EXCLUDING the edge sample).
fn reflectPad1HW(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const h = shp[2];
    const w = shp[3];
    const top = try sliceAxis(x, 2, 1, 2, s);
    defer _ = mlx.mlx_array_free(top);
    const bot = try sliceAxis(x, 2, h - 2, h - 1, s);
    defer _ = mlx.mlx_array_free(bot);
    const xv = try concat(&[_]mlx.mlx_array{ top, x, bot }, 2, s);
    defer _ = mlx.mlx_array_free(xv);
    const left = try sliceAxis(xv, 3, 1, 2, s);
    defer _ = mlx.mlx_array_free(left);
    const right = try sliceAxis(xv, 3, w - 2, w - 1, s);
    defer _ = mlx.mlx_array_free(right);
    return concat(&[_]mlx.mlx_array{ left, xv, right }, 3, s);
}

/// Reflect-pad bottom+right by 1 (the reference Downsample3D's (0,1,0,1) pad).
fn reflectPadBR(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const h = shp[2];
    const w = shp[3];
    const bot = try sliceAxis(x, 2, h - 2, h - 1, s);
    defer _ = mlx.mlx_array_free(bot);
    const xv = try concat(&[_]mlx.mlx_array{ x, bot }, 2, s);
    defer _ = mlx.mlx_array_free(xv);
    const right = try sliceAxis(xv, 3, w - 2, w - 1, s);
    defer _ = mlx.mlx_array_free(right);
    return concat(&[_]mlx.mlx_array{ xv, right }, 3, s);
}

/// CAUSAL temporal padding: `n` ZERO frames at the FRONT only, on
/// [1, T, H, W, C]. Front-only is the whole point — a symmetric pad would let
/// a frame see its own future, which is what makes the chunked encode
/// inconsistent with a single pass.
fn causalPadFront(x: mlx.mlx_array, n: c_int, s: S) !mlx.mlx_array {
    if (n == 0) return contig(x, s);
    const shp = mlx.getShape(x);
    const zshape = [_]c_int{ shp[0], n, shp[2], shp[3], shp[4] };
    var z = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_zeros(&z, &zshape, zshape.len, mlx.mlx_array_dtype(x), s));
    return concat(&[_]mlx.mlx_array{ z, x }, 1, s);
}

fn conv3dBias(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, st: c_int, ss: c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_conv3d(&o, xc, w, st, ss, ss, 0, 0, 0, 1, 1, 1, 1, s));
    return addA(o, b, s);
}

/// The reference's `TemporalIsolatedGroupNorm`: GroupNorm(32, eps 1e-6,
/// affine) with statistics computed PER FRAME (time folded into the batch), on
/// [1, T, H, W, C]. Sharing statistics across frames is the obvious-looking
/// simplification and it is wrong — it couples frames the chunked encode has
/// already decided are independent.
fn groupNorm32(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const t = shp[1];
    const h = shp[2];
    const w_ = shp[3];
    const c = shp[4];
    const g = try reshape(x, &[_]c_int{ 1, t, h, w_, 32, @divExact(c, 32) }, s);
    defer _ = mlx.mlx_array_free(g);
    // NOT axis 1: the statistics are per (batch, FRAME, group).
    const axes = [_]c_int{ 2, 3, 5 };
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axes(&mean, g, &axes, axes.len, true, s));
    const diff = try subA(g, mean, s);
    defer _ = mlx.mlx_array_free(diff);
    const sq = try mulA(diff, diff, s);
    defer _ = mlx.mlx_array_free(sq);
    var vr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vr);
    try mlx.check(mlx.mlx_mean_axes(&vr, sq, &axes, axes.len, true, s));
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    const ve = try addA(vr, eps, s);
    defer _ = mlx.mlx_array_free(ve);
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, ve, s));
    const nrm = try mulA(diff, rs, s);
    defer _ = mlx.mlx_array_free(nrm);
    const back = try reshape(nrm, &[_]c_int{ 1, t, h, w_, c }, s);
    defer _ = mlx.mlx_array_free(back);
    const sc = try mulA(back, w, s);
    defer _ = mlx.mlx_array_free(sc);
    return addA(sc, b, s);
}

const EncRes = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    conv1_w: mlx.mlx_array,
    conv1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    conv2_w: mlx.mlx_array,
    conv2_b: mlx.mlx_array,
    /// 1x1 shortcut as a pre-transposed [in, out] matmul weight; null ctx when
    /// in == out.
    nin_wt: mlx.mlx_array = .{ .ctx = null },
    nin_b: mlx.mlx_array = .{ .ctx = null },

    fn deinit(self: *EncRes) void {
        for ([_]mlx.mlx_array{ self.norm1_w, self.norm1_b, self.conv1_w, self.conv1_b, self.norm2_w, self.norm2_b, self.conv2_w, self.conv2_b }) |a2| _ = mlx.mlx_array_free(a2);
        if (self.nin_wt.ctx != null) _ = mlx.mlx_array_free(self.nin_wt);
        if (self.nin_b.ctx != null) _ = mlx.mlx_array_free(self.nin_b);
    }
};

const EncLevel = struct {
    blocks: [ENC_RES_BLOCKS]EncRes,
    down_w: mlx.mlx_array = .{ .ctx = null },
    down_b: mlx.mlx_array = .{ .ctx = null },

    fn deinit(self: *EncLevel) void {
        for (&self.blocks) |*bk| bk.deinit();
        if (self.down_w.ctx != null) _ = mlx.mlx_array_free(self.down_w);
        if (self.down_b.ctx != null) _ = mlx.mlx_array_free(self.down_b);
    }
};

pub const Encoder = struct {
    allocator: std.mem.Allocator,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    levels: [ENC_LEVELS]EncLevel,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    quant_wt: mlx.mlx_array, // [48, 48] pre-transposed [in, out]
    quant_b: mlx.mlx_array,
    latents_mean: mlx.mlx_array,
    latents_std: mlx.mlx_array,

    /// [O, I, kt, kh, kw] -> f32 [O, kt, kh, kw, I], the layout mlx_conv3d
    /// wants. Kept in FULL (not tap-truncated) so one weight set serves both
    /// the keyframe and the reference-video paths.
    fn convTap(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
        const name = try std.fmt.allocPrint(a, fmt, args);
        defer a.free(name);
        const raw = try ownWeight(w, name);
        defer _ = mlx.mlx_array_free(raw);
        const tr = try transpose(raw, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
        defer _ = mlx.mlx_array_free(tr);
        const trc = try contig(tr, s);
        defer _ = mlx.mlx_array_free(trc);
        return astype(trc, mlx.mlx_dtype.float32, s);
    }

    fn vec1(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
        const name = try std.fmt.allocPrint(a, fmt, args);
        defer a.free(name);
        const raw = try ownWeight(w, name);
        defer _ = mlx.mlx_array_free(raw);
        return astype(raw, mlx.mlx_dtype.float32, s);
    }

    /// 1x1x1 conv weight [O, I, 1, 1, 1] -> pre-transposed f32 [I, O].
    fn oneByOne(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
        const name = try std.fmt.allocPrint(a, fmt, args);
        defer a.free(name);
        const raw = try ownWeight(w, name);
        defer _ = mlx.mlx_array_free(raw);
        const shp = mlx.getShape(raw);
        const sq = try reshape(raw, &[_]c_int{ shp[0], shp[1] }, s);
        defer _ = mlx.mlx_array_free(sq);
        const tr = try transpose(sq, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(tr);
        const trc = try contig(tr, s);
        defer _ = mlx.mlx_array_free(trc);
        return astype(trc, mlx.mlx_dtype.float32, s);
    }

    pub fn load(allocator: std.mem.Allocator, w: *const Weights, s: S) !Encoder {
        var self: Encoder = undefined;
        self.allocator = allocator;
        self.conv_in_w = try convTap(w, allocator, "encoder.conv_in.weight", .{}, s);
        self.conv_in_b = try vec1(w, allocator, "encoder.conv_in.bias", .{}, s);
        for (0..ENC_LEVELS) |lv| {
            var level: EncLevel = .{ .blocks = undefined };
            for (0..ENC_RES_BLOCKS) |bi| {
                var blk: EncRes = .{
                    .norm1_w = try vec1(w, allocator, "encoder.down.{d}.block.{d}.norm1.weight", .{ lv, bi }, s),
                    .norm1_b = try vec1(w, allocator, "encoder.down.{d}.block.{d}.norm1.bias", .{ lv, bi }, s),
                    .conv1_w = try convTap(w, allocator, "encoder.down.{d}.block.{d}.conv1.weight", .{ lv, bi }, s),
                    .conv1_b = try vec1(w, allocator, "encoder.down.{d}.block.{d}.conv1.bias", .{ lv, bi }, s),
                    .norm2_w = try vec1(w, allocator, "encoder.down.{d}.block.{d}.norm2.weight", .{ lv, bi }, s),
                    .norm2_b = try vec1(w, allocator, "encoder.down.{d}.block.{d}.norm2.bias", .{ lv, bi }, s),
                    .conv2_w = try convTap(w, allocator, "encoder.down.{d}.block.{d}.conv2.weight", .{ lv, bi }, s),
                    .conv2_b = try vec1(w, allocator, "encoder.down.{d}.block.{d}.conv2.bias", .{ lv, bi }, s),
                };
                // in != out only on the first block of a widening level.
                const in_ch = if (bi == 0) encBlockIn(lv) else encBlockMid(lv);
                if (in_ch != encBlockMid(lv)) {
                    blk.nin_wt = try oneByOne(w, allocator, "encoder.down.{d}.block.{d}.nin_shortcut.weight", .{ lv, bi }, s);
                    blk.nin_b = try vec1(w, allocator, "encoder.down.{d}.block.{d}.nin_shortcut.bias", .{ lv, bi }, s);
                }
                level.blocks[bi] = blk;
            }
            if (encHasDown(lv)) {
                level.down_w = try convTap(w, allocator, "encoder.down.{d}.downsample.conv.weight", .{lv}, s);
                level.down_b = try vec1(w, allocator, "encoder.down.{d}.downsample.conv.bias", .{lv}, s);
            }
            self.levels[lv] = level;
        }
        self.norm_out_w = try vec1(w, allocator, "encoder.norm_out.weight", .{}, s);
        self.norm_out_b = try vec1(w, allocator, "encoder.norm_out.bias", .{}, s);
        self.conv_out_w = try convTap(w, allocator, "encoder.conv_out.weight", .{}, s);
        self.conv_out_b = try vec1(w, allocator, "encoder.conv_out.bias", .{}, s);
        self.quant_wt = try oneByOne(w, allocator, "quant_conv.weight", .{}, s);
        self.quant_b = try vec1(w, allocator, "quant_conv.bias", .{}, s);
        self.latents_mean = try vec1(w, allocator, "latents_mean", .{}, s);
        self.latents_std = try vec1(w, allocator, "latents_std", .{}, s);
        return self;
    }

    pub fn deinit(self: *Encoder) void {
        for ([_]mlx.mlx_array{
            self.conv_in_w,  self.conv_in_b,  self.norm_out_w,   self.norm_out_b,
            self.conv_out_w, self.conv_out_b, self.quant_wt,     self.quant_b,
            self.latents_mean, self.latents_std,
        }) |a2| _ = mlx.mlx_array_free(a2);
        for (&self.levels) |*lv| lv.deinit();
    }

    /// One CausalConv3d with `padding=1` on every axis: reflect in space,
    /// ZERO and FRONT-ONLY in time.
    fn causalConv(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sp = try reflectPad1HW(x, s);
        defer _ = mlx.mlx_array_free(sp);
        const tp = try causalPadFront(sp, 2, s);
        defer _ = mlx.mlx_array_free(tp);
        return conv3dBias(tp, w, b, 1, 1, s);
    }

    fn resBlock(self: *const Encoder, blk: *const EncRes, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        _ = self;
        const n1 = try groupNorm32(x, blk.norm1_w, blk.norm1_b, s);
        defer _ = mlx.mlx_array_free(n1);
        const a1 = try siluA(n1, s);
        defer _ = mlx.mlx_array_free(a1);
        const h1 = try causalConv(a1, blk.conv1_w, blk.conv1_b, s);
        defer _ = mlx.mlx_array_free(h1);
        const n2 = try groupNorm32(h1, blk.norm2_w, blk.norm2_b, s);
        defer _ = mlx.mlx_array_free(n2);
        const a2 = try siluA(n2, s);
        defer _ = mlx.mlx_array_free(a2);
        const h2 = try causalConv(a2, blk.conv2_w, blk.conv2_b, s);
        defer _ = mlx.mlx_array_free(h2);
        var sc = x;
        var sc_owned = false;
        if (blk.nin_wt.ctx != null) {
            // A 1x1x1 conv is a per-position linear on the channel axis, so it
            // needs no padding and works at any rank.
            sc = try linT(x, blk.nin_wt, blk.nin_b, s);
            sc_owned = true;
        }
        defer if (sc_owned) {
            _ = mlx.mlx_array_free(sc);
        };
        return addA(h2, sc, s);
    }

    /// One tile [1, T, th, tw, 3] (already pixel-normalized) -> moments
    /// [1, 48, T_lat, th/16, tw/16].
    fn encodeMoments(self: *const Encoder, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var h = try causalConv(x, self.conv_in_w, self.conv_in_b, s);
        errdefer _ = mlx.mlx_array_free(h);
        for (0..ENC_LEVELS) |lv| {
            const level = &self.levels[lv];
            for (0..ENC_RES_BLOCKS) |bi| {
                const nh = try self.resBlock(&level.blocks[bi], h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (level.down_w.ctx != null) {
                // Downsample3D: reflect-pad BOTTOM+RIGHT only (never all four
                // sides), then a causal conv with the level's own strides. Its
                // padding is (1,0,0) — temporal only, no spatial pad here.
                const pd = if (ENC_SPACE_DOWN[lv] == 2) try reflectPadBR(h, s) else try contig(h, s);
                defer _ = mlx.mlx_array_free(pd);
                const tp = try causalPadFront(pd, 2, s);
                defer _ = mlx.mlx_array_free(tp);
                const nh = try conv3dBias(tp, level.down_w, level.down_b, @intCast(ENC_TIME_DOWN[lv]), @intCast(ENC_SPACE_DOWN[lv]), s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
        }
        // `h` owns the live value throughout; the errdefer above frees it
        // exactly once on any error path.
        const step = struct {
            fn adv(cur: *mlx.mlx_array, next: mlx.mlx_array) void {
                _ = mlx.mlx_array_free(cur.*);
                cur.* = next;
            }
        }.adv;
        step(&h, try groupNorm32(h, self.norm_out_w, self.norm_out_b, s));
        step(&h, try siluA(h, s));
        step(&h, try causalConv(h, self.conv_out_w, self.conv_out_b, s));
        step(&h, try linT(h, self.quant_wt, self.quant_b, s));
        // [1, T_lat, lh, lw, 48] -> [1, 48, T_lat, lh, lw]
        step(&h, try transpose(h, &[_]c_int{ 0, 4, 1, 2, 3 }, s));
        const out = try contig(h, s);
        _ = mlx.mlx_array_free(h);
        return out;
    }

    /// Single keyframe image -> NORMALIZED latent.
    /// `pixels` [1, 3, 1, H, W] f32 in [-1, 1] at the target canvas;
    /// returns [1, 24, 1, H/16, W/16] f32.
    ///
    /// Spatially TILED above the 256-px tile extent like the decoder — and for
    /// the same reason: tiles renormalize their rope... no rope here, but the
    /// reference encodes tiled ALWAYS (`tiling=True`), blending MOMENTS at
    /// latent granularity against the RAW neighbours, so parity means doing
    /// the same.
    pub fn encodeImage(self: *const Encoder, pixels: mlx.mlx_array, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
        return self.encodeVideo(pixels, alloc, s);
    }

    /// `pixels` [1, 3, T, H, W] f32 in [-1, 1] -> NORMALIZED latents
    /// [1, 24, T_lat, H/16, W/16].
    ///
    /// T == 1 is the fl2va keyframe (T_lat 1); T > 1 is a ref2va reference
    /// video, encoded in 17-FRAME CLIPS with the tail repeat-padded and the
    /// last `TOKEN_DROP` latent frames discarded. `videoLatentT` is the same
    /// ladder read from the other end, so a caller that snapped its frame
    /// count to 17k+5 gets exactly `5k+2` latent frames back.
    pub fn encodeVideo(self: *const Encoder, pixels: mlx.mlx_array, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
        const pshp = mlx.getShape(pixels);
        const pt: u32 = @intCast(pshp[2]);
        const ph: u32 = @intCast(pshp[3]);
        const pw: u32 = @intCast(pshp[4]);

        // [1,3,T,H,W] -> [1,T,H,W,3], then [-1,1] -> ImageNet normalization.
        const t5 = try transpose(pixels, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
        defer _ = mlx.mlx_array_free(t5);
        const t5c = try contig(t5, s);
        defer _ = mlx.mlx_array_free(t5c);
        const f32x = try astype(t5c, mlx.mlx_dtype.float32, s);
        defer _ = mlx.mlx_array_free(f32x);
        const half = mlx.mlx_array_new_float(0.5);
        defer _ = mlx.mlx_array_free(half);
        const unit01a = try mulA(f32x, half, s);
        defer _ = mlx.mlx_array_free(unit01a);
        const unit01 = try addA(unit01a, half, s);
        defer _ = mlx.mlx_array_free(unit01);
        const msh = [_]c_int{ 1, 1, 1, 1, 3 };
        const pm = mlx.mlx_array_new_data(&PIXEL_MEAN, &msh, 5, mlx.mlx_dtype.float32);
        defer _ = mlx.mlx_array_free(pm);
        const psd = mlx.mlx_array_new_data(&PIXEL_STD, &msh, 5, mlx.mlx_dtype.float32);
        defer _ = mlx.mlx_array_free(psd);
        const cen = try subA(unit01, pm, s);
        defer _ = mlx.mlx_array_free(cen);
        const nthwc = try divA(cen, psd, s);
        defer _ = mlx.mlx_array_free(nthwc);

        const moments = if (pt == 1)
            try self.adaptiveEncode(nthwc, ph, pw, alloc, s)
        else
            try self.encodeTemporal(nthwc, pt, ph, pw, alloc, s);
        defer _ = mlx.mlx_array_free(moments);

        // mean = first 24 channels; normalize by the latent statistics.
        const mean24 = try sliceAxis(moments, 1, 0, 24, s);
        defer _ = mlx.mlx_array_free(mean24);
        const csh = [_]c_int{ 1, 24, 1, 1, 1 };
        const lm = try reshape(self.latents_mean, &csh, s);
        defer _ = mlx.mlx_array_free(lm);
        const ls = try reshape(self.latents_std, &csh, s);
        defer _ = mlx.mlx_array_free(ls);
        const centered = try subA(mean24, lm, s);
        defer _ = mlx.mlx_array_free(centered);
        return divA(centered, ls, s);
    }

    /// Single pass or spatially tiled, per the reference's `_adaptive_encode`.
    fn adaptiveEncode(self: *const Encoder, x: mlx.mlx_array, ph: u32, pw: u32, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
        if (fitsSingleTile(ph) and fitsSingleTile(pw)) return self.encodeMoments(x, s);
        return self.encodeTiled(x, alloc, s);
    }

    /// The reference's `encode_temporal`. The 17-frame clip is what the VAE was
    /// TRAINED on, so encoding a long video in one pass is not an optimization
    /// that was skipped — it is a different (wrong) computation. The tail is
    /// repeat-padded with the LAST frame, never zero-padded or truncated.
    fn encodeTemporal(self: *const Encoder, x: mlx.mlx_array, pt: u32, ph: u32, pw: u32, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
        var padded = try contig(x, s);
        defer _ = mlx.mlx_array_free(padded);
        const rem = pt % CLIP_LENGTH;
        var total = pt;
        if (rem != 0) {
            const pad_n = CLIP_LENGTH - rem;
            const last = try sliceAxis(padded, 1, @intCast(pt - 1), @intCast(pt), s);
            defer _ = mlx.mlx_array_free(last);
            var parts = try alloc.alloc(mlx.mlx_array, pad_n + 1);
            defer alloc.free(parts);
            parts[0] = padded;
            for (1..pad_n + 1) |i| parts[i] = last;
            const cat = try concat(parts, 1, s);
            _ = mlx.mlx_array_free(padded);
            padded = cat;
            total = pt + pad_n;
        }

        const chunks = total / CLIP_LENGTH;
        var outs = try alloc.alloc(mlx.mlx_array, chunks);
        defer alloc.free(outs);
        var built: usize = 0;
        defer for (outs[0..built]) |o| {
            _ = mlx.mlx_array_free(o);
        };
        for (0..chunks) |i| {
            const lo: c_int = @intCast(i * CLIP_LENGTH);
            const clip = try sliceAxis(padded, 1, lo, lo + @as(c_int, @intCast(CLIP_LENGTH)), s);
            defer _ = mlx.mlx_array_free(clip);
            outs[i] = try self.adaptiveEncode(clip, ph, pw, alloc, s);
            built += 1;
        }
        const cat = if (chunks == 1) try contig(outs[0], s) else try concat(outs, 2, s);
        defer _ = mlx.mlx_array_free(cat);
        // Drop the trailing TOKEN_DROP latent frames — the tail the repeat-pad
        // fabricated, plus the clip's own causal warm-up.
        const n_lat: c_int = mlx.getShape(cat)[2];
        return sliceAxis(cat, 2, 0, n_lat - @as(c_int, @intCast(TOKEN_DROP)), s);
    }

    /// Reference `tiled_encode`: encode raw pixel tiles, blend each against its
    /// RAW up/left neighbour at LATENT granularity, trim, assemble.
    fn encodeTiled(self: *const Encoder, nhwc: mlx.mlx_array, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
        const shp = mlx.getShape(nhwc);
        const ph: u32 = @intCast(shp[2]);
        const pw: u32 = @intCast(shp[3]);
        var yp = try splitTiles(alloc, ph);
        defer yp.deinit();
        var xp = try splitTiles(alloc, pw);
        defer xp.deinit();
        const ny = yp.count();
        const nx = xp.count();

        var raw = try alloc.alloc(mlx.mlx_array, ny * nx);
        defer alloc.free(raw);
        var built: usize = 0;
        defer for (raw[0..built]) |r| {
            _ = mlx.mlx_array_free(r);
        };
        for (yp.starts, 0..) |y0, i| {
            for (xp.starts, 0..) |x0, j| {
                // Input is [1, T, H, W, C]: the spatial axes are 2 and 3.
                const th = try sliceAxis(nhwc, 2, @intCast(y0), @intCast(y0 + yp.len), s);
                defer _ = mlx.mlx_array_free(th);
                const tile = try sliceAxis(th, 3, @intCast(x0), @intCast(x0 + xp.len), s);
                defer _ = mlx.mlx_array_free(tile);
                raw[i * nx + j] = try self.encodeMoments(tile, s);
                built += 1;
            }
        }

        var rows = try alloc.alloc(mlx.mlx_array, ny);
        defer alloc.free(rows);
        var rows_built: usize = 0;
        errdefer for (rows[0..rows_built]) |r| {
            _ = mlx.mlx_array_free(r);
        };
        for (0..ny) |i| {
            var cols = try alloc.alloc(mlx.mlx_array, nx);
            defer alloc.free(cols);
            var cols_built: usize = 0;
            errdefer for (cols[0..cols_built]) |c2| {
                _ = mlx.mlx_array_free(c2);
            };
            for (0..nx) |j| {
                var tile = try contig(raw[i * nx + j], s);
                errdefer _ = mlx.mlx_array_free(tile);
                if (i > 0) {
                    const b = try blendAxis(raw[(i - 1) * nx + j], tile, yp.overlaps[i - 1] / VAE_RATIO, 3, alloc, mlx.mlx_dtype.float32, s);
                    _ = mlx.mlx_array_free(tile);
                    tile = b;
                }
                if (j > 0) {
                    const b = try blendAxis(raw[i * nx + (j - 1)], tile, xp.overlaps[j - 1] / VAE_RATIO, 4, alloc, mlx.mlx_dtype.float32, s);
                    _ = mlx.mlx_array_free(tile);
                    tile = b;
                }
                if (i + 1 < ny) {
                    const h2: c_int = mlx.getShape(tile)[3];
                    const t2 = try sliceAxis(tile, 3, 0, h2 - @as(c_int, @intCast(yp.overlaps[i] / VAE_RATIO)), s);
                    _ = mlx.mlx_array_free(tile);
                    tile = t2;
                }
                if (j + 1 < nx) {
                    const w2: c_int = mlx.getShape(tile)[4];
                    const t2 = try sliceAxis(tile, 4, 0, w2 - @as(c_int, @intCast(xp.overlaps[j] / VAE_RATIO)), s);
                    _ = mlx.mlx_array_free(tile);
                    tile = t2;
                }
                cols[j] = tile;
                cols_built += 1;
            }
            rows[i] = try concat(cols, 4, s);
            rows_built += 1;
            for (cols) |c2| _ = mlx.mlx_array_free(c2);
        }
        const outp = try concat(rows, 3, s);
        for (rows) |r| _ = mlx.mlx_array_free(r);
        rows_built = 0;
        return outp;
    }
};

test "minimax h3 vae: encoder geometry ladder matches the reference config" {
    // block_in = [mid[0]] ++ mid[:-1]; mid = ch * ch_mult.
    const want_in = [_]u32{ 128, 128, 256, 256, 512, 512 };
    const want_mid = [_]u32{ 128, 256, 256, 512, 512, 1024 };
    for (0..ENC_LEVELS) |lv| {
        try testing.expectEqual(want_in[lv], encBlockIn(lv));
        try testing.expectEqual(want_mid[lv], encBlockMid(lv));
    }
    // Downsamples on levels 0-3 only (space*time > 1); the spatial strides
    // multiply to the 16x VAE ratio.
    var space: u32 = 1;
    for (0..ENC_LEVELS) |lv| {
        try testing.expectEqual(lv < 4, encHasDown(lv));
        space *= ENC_SPACE_DOWN[lv];
    }
    try testing.expectEqual(VAE_RATIO, space);
}

test "minimax h3 vae live: encoder parity vs the executed reference" {
    // Fixture from tests/dump_minimax_h3_vae_encoder_fixture.py, which RUNS
    // the reference conv encoder (plain torch — unlike the DiT, nothing here
    // needs comfy_kitchen). Four cases: a single-tile 128px image, a TILED
    // 384px one (so the moment-blend assembly is pinned against the
    // reference's own tiled_encode output), and two VIDEOS — 5 frames (one
    // 17-frame clip after repeat-padding) and 22 (TWO clips, the only case
    // that exercises the seam between independently-encoded clips).
    //
    // The video frames MOVE, so a port that collapses, reverses or mis-strides
    // the temporal axis cannot pass; a static clip would let all three through.
    const raw_model = std.c.getenv("MINIMAX_H3_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("MINIMAX_H3_VAE_ENC_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    // An EMPTY env var must skip like an absent one: getenv("")-style unset
    // (`VAR= binary`) reaches here as "", and load_safetensors("") is an
    // uncatchable MLX error that killed a whole live-test run.
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();

    const vae_path = try std.fmt.allocPrint(a, "{s}/video_vae.safetensors", .{model_dir});
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var enc = try Encoder.load(a, &vw, s);
    defer enc.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    for ([_][2][]const u8{
        .{ "x", "latent" },
        .{ "x_tiled", "latent_tiled" },
        .{ "v5", "latent_v5" },
        .{ "v22", "latent_v22" },
    }) |pair| {
        const x = try ownWeight(&fx, pair[0]);
        defer _ = mlx.mlx_array_free(x);
        const want = try ownWeight(&fx, pair[1]);
        defer _ = mlx.mlx_array_free(want);
        const got = try enc.encodeVideo(x, a, s);
        defer _ = mlx.mlx_array_free(got);
        // Shape first: a temporal ladder that is one token off still scores a
        // high cosine on the overlapping rows, and cosineSimV would compare
        // different element counts.
        const gs = mlx.getShape(got);
        const ws = mlx.getShape(want);
        testing.expectEqual(ws.len, gs.len) catch |e| {
            std.debug.print("[h3-vae-enc] {s}: rank {d} vs {d}\n", .{ pair[0], gs.len, ws.len });
            return e;
        };
        for (ws, gs, 0..) |wv, gv, i| {
            testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("[h3-vae-enc] {s}: axis {d} = {d}, want {d}\n", .{ pair[0], i, gv, wv });
                return e;
            };
        }
        const cos = try cosineSimV(got, want, s);
        std.debug.print("[h3-vae-enc] {s}: shape ok, cos={d:.6}\n", .{ pair[0], cos });
        try testing.expect(cos > 0.999);
    }
}

test "minimax h3 vae: the encode ladder is videoLatentT read backwards" {
    // encodeTemporal produces ceil(T/17)*TOKENS_CHUNK_SIZE - TOKEN_DROP latent
    // frames. A caller that snapped its frame count to the 17k+5 ladder must
    // get exactly what `videoLatentT` promises the DiT — the two are computed
    // by different code and a disagreement is a shape error minutes into a
    // generation, not at the call.
    for ([_]u32{ 5, 22, 39, 56, 124, 209, 362 }) |frames| {
        const chunks = (frames + CLIP_LENGTH - 1) / CLIP_LENGTH;
        const encoded = chunks * TOKENS_CHUNK_SIZE - TOKEN_DROP;
        try testing.expectEqual(h3.videoLatentT(frames), encoded);
        // and every case really is ON the ladder, so the equality above is not
        // being checked against counts the server would never produce
        try testing.expectEqual(@as(u32, 5), frames % CLIP_LENGTH);
    }
}

fn cosineSimV(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: S) !f32 {
    const af = try astype(a_arr, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(af);
    const bf = try astype(b_arr, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const ab = try mulA(af, bf, s);
    defer _ = mlx.mlx_array_free(ab);
    const aa = try mulA(af, af, s);
    defer _ = mlx.mlx_array_free(aa);
    const bb = try mulA(bf, bf, s);
    defer _ = mlx.mlx_array_free(bb);
    var dot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dot);
    var na = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(na);
    var nb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb);
    try mlx.check(mlx.mlx_sum(&dot, ab, false, s));
    try mlx.check(mlx.mlx_sum(&na, aa, false, s));
    try mlx.check(mlx.mlx_sum(&nb, bb, false, s));
    try mlx.check(mlx.mlx_array_eval(dot));
    var d: f32 = 0;
    var x1: f32 = 0;
    var x2: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&d, dot));
    try mlx.check(mlx.mlx_array_item_float32(&x1, na));
    try mlx.check(mlx.mlx_array_item_float32(&x2, nb));
    return d / (@sqrt(x1) * @sqrt(x2) + 1e-12);
}

test "minimax h3 vae live: encode->decode round trip preserves orientation" {
    // fl2va's first live run produced a MIRRORED first frame (left/right means
    // inverted vs the keyframe). The encoder is pinned against the executed
    // reference, but the decoder never was — and a W-flip in a video decoder
    // is invisible on organic content (a cat at a window mirrors cleanly).
    // A pure encode->decode round trip of a hard left/right split image
    // discriminates: mirrored output = VAE-side flip; clean output = the flip
    // lives in the DiT cond wiring instead.
    const raw_model = std.c.getenv("MINIMAX_H3_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const a = testing.allocator;
    const s = mlx.gpuStream();

    const vae_path = try std.fmt.allocPrint(a, "{s}/video_vae.safetensors", .{model_dir});
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var enc = try Encoder.load(a, &vw, s);
    defer enc.deinit();
    var dec = try Decoder.load(a, &vw, .{}, mlx.mlx_dtype.bfloat16, s);
    defer dec.deinit();

    // [1,3,1,128,128] in [-1,1]: left dark (-0.9), right bright (+0.9), and a
    // dark TOP band (rows 0..16) so the H axis is checked too.
    const px = 128;
    const buf = try a.alloc(f32, 3 * px * px);
    defer a.free(buf);
    for (0..px) |y| {
        for (0..px) |x| {
            var v: f32 = if (x < px / 2) -0.9 else 0.9;
            if (y < 16) v = -0.9;
            for (0..3) |c| buf[c * px * px + y * px + x] = v;
        }
    }
    const shape5 = [_]c_int{ 1, 3, 1, px, px };
    const img = mlx.mlx_array_new_data(buf.ptr, &shape5, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(img);

    const lat = try enc.encodeImage(img, a, s);
    defer _ = mlx.mlx_array_free(lat);
    const pixels = try decode(&dec, lat, a, s);
    defer _ = mlx.mlx_array_free(pixels);
    const pf = try astype(pixels, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(pf);
    try mlx.check(mlx.mlx_array_eval(pf));

    const shp = mlx.getShape(pf);
    const oh: usize = @intCast(shp[3]);
    const ow: usize = @intCast(shp[4]);
    const data = mlx.mlx_array_data_float32(pf) orelse return error.NoPixelData;
    // Means over frame 0, channel 0, below the top band.
    var left: f64 = 0;
    var right: f64 = 0;
    var top: f64 = 0;
    var bottom: f64 = 0;
    var nl: f64 = 0;
    var nt: f64 = 0;
    for (0..oh) |y| {
        for (0..ow) |x| {
            const v: f64 = data[y * ow + x];
            if (y >= 24) {
                if (x < ow / 2) {
                    left += v;
                    nl += 1;
                } else right += v;
            }
            if (x >= ow / 2) {
                if (y < 8) {
                    top += v;
                    nt += 1;
                } else if (y >= 24) bottom += v;
            }
        }
    }
    const lm = left / nl;
    const rm = right / nl;
    const tm = top / nt;
    const bm = bottom / (nl - nt + 1);
    std.debug.print("[h3-vae-rt] left={d:.3} right={d:.3} top={d:.3} bottomish={d:.3}\n", .{ lm, rm, tm, bm });
    // Input: left dark, right bright, top dark. A mirror flips the sign.
    try testing.expect(rm - lm > 0.5);
    try testing.expect(tm < rm - 0.5);
}

test "minimax h3 vae live: the Latent2RGB preview resembles the decoded frame" {
    // The fixture oracle in preview.zig proves our projection IS ComfyUI's.
    // It cannot prove ComfyUI's projection looks like the video, which is the
    // whole point of a preview (issue #208 review: "no oracle or perceptual
    // test proving it looks like anything"). So: encode a real image with the
    // real VAE, decode it back, box-average the decode down to the latent grid
    // and correlate. The GOLDEN-ANGLE HUE WHEEL this file used to ship is the
    // control arm — a bar the retired projection also clears proves nothing.
    const raw_model = std.c.getenv("MINIMAX_H3_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    if (model_dir.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();

    const vae_path = try std.fmt.allocPrint(a, "{s}/video_vae.safetensors", .{model_dir});
    defer a.free(vae_path);
    // The VAE is the only weight file this test needs, so a dir that has it is
    // enough: a missing one SKIPS rather than failing an unrelated pack.
    const io = std.Io.Threaded.global_single_threaded.io();
    if (std.Io.Dir.openFileAbsolute(io, vae_path, .{})) |f| f.close(io) else |_| return error.SkipZigTest;
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var enc = try Encoder.load(a, &vw, s);
    defer enc.deinit();
    var dec = try Decoder.load(a, &vw, .{}, mlx.mlx_dtype.bfloat16, s);
    defer dec.deinit();

    // One independent colour per LATENT CELL (H3 compresses 16x spatially), so
    // the bar is 3 real predictions per cell rather than a gradient any linear
    // map reproduces.
    const px: u32 = 256;
    const buf = try preview.perceptualTestFrame(a, px, VAE_RATIO);
    defer a.free(buf);
    const shape5 = [_]c_int{ 1, 3, 1, @intCast(px), @intCast(px) };
    const img = mlx.mlx_array_new_data(buf.ptr, &shape5, 5, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(img);

    const lat = try enc.encodeImage(img, a, s);
    defer _ = mlx.mlx_array_free(lat);
    const latf = try astype(lat, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(latf);
    try mlx.check(mlx.mlx_array_eval(latf));
    const lshp = mlx.getShape(latf);
    const lc: u32 = @intCast(lshp[1]);
    const lt: u32 = @intCast(lshp[2]);
    const lh: u32 = @intCast(lshp[3]);
    const lw: u32 = @intCast(lshp[4]);
    try testing.expectEqual(preview.minimax_h3.channels(), lc);
    const ldata = mlx.mlx_array_data_float32(latf) orelse return error.NoPixelData;
    const lat_host = ldata[0 .. @as(usize, lc) * lt * lh * lw];

    // The preview the client would see for this latent, at latent resolution.
    const fit = try a.alloc(u8, @as(usize, lh) * lw * 3);
    defer a.free(fit);
    preview.latentSliceToRgb(preview.minimax_h3, lat_host, lt, lh, lw, 0, fit);
    const ctrl_map = preview.goldenAngleControlMap(24);
    const ctrl = try a.alloc(u8, @as(usize, lh) * lw * 3);
    defer a.free(ctrl);
    preview.latentSliceToRgb(ctrl_map, lat_host, lt, lh, lw, 0, ctrl);

    // What the user eventually gets: the same latent through the real decoder.
    const pixels = try decode(&dec, lat, a, s);
    defer _ = mlx.mlx_array_free(pixels);
    const pf = try astype(pixels, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(pf);
    try mlx.check(mlx.mlx_array_eval(pf));
    const pshp = mlx.getShape(pf);
    const ot: usize = @intCast(pshp[2]);
    const oh: usize = @intCast(pshp[3]);
    const ow: usize = @intCast(pshp[4]);
    const pdata = mlx.mlx_array_data_float32(pf) orelse return error.NoPixelData;
    const decoded = try a.alloc(u8, oh * ow * 3);
    defer a.free(decoded);
    for (0..oh) |y| {
        for (0..ow) |x| {
            for (0..3) |c| {
                const v = pdata[c * ot * oh * ow + y * ow + x];
                const u = (v + 1.0) * 0.5 * 255.0;
                decoded[(y * ow + x) * 3 + c] = @intFromFloat(@min(255.0, @max(0.0, u)));
            }
        }
    }
    const decoded_small = try preview.boxDownsampleRgb(a, decoded, @intCast(ow), @intCast(oh), lw, lh);
    defer a.free(decoded_small);

    const corr_fit = preview.rgbCorrelation(fit, decoded_small);
    const corr_ctrl = preview.rgbCorrelation(ctrl, decoded_small);
    const chroma_fit = preview.rgbChromaCorrelation(fit, decoded_small);
    const chroma_ctrl = preview.rgbChromaCorrelation(ctrl, decoded_small);
    std.debug.print(
        "[h3-preview] latent={d}x{d}x{d} decode={d}x{d}x{d} corr_fit={d:.4} corr_huewheel={d:.4} chroma_fit={d:.4} chroma_huewheel={d:.4}\n",
        .{ lc, lh, lw, ot, oh, ow, corr_fit, corr_ctrl, chroma_fit, chroma_ctrl },
    );
    // The published fit tracks the decode; the hue wheel does not. All four
    // numbers are the assertion: an absolute bar alone passes on a metric any
    // linear map clears, a relative one alone passes on two bad maps, and full
    // RGB alone is mostly luminance — which is the half a hue wheel gets free.
    // Measured 2026-08-30 (FL2VA 8-bit, M-series): fit 0.740 / chroma 0.880,
    // control 0.029 / 0.042. The absolute floors are sanity bounds; the CONTROL
    // margin is the discriminator, and it sits near 0.7 with the bar at 0.3.
    try testing.expect(corr_fit > 0.6);
    try testing.expect(corr_fit > corr_ctrl + 0.3);
    try testing.expect(chroma_fit > 0.7);
    try testing.expect(chroma_fit > chroma_ctrl + 0.3);
}
