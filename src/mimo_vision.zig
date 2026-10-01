//! MiMo-ViT: the MiMo-V2.6 vision tower and its image preprocessing.
//!
//! Preprocessing follows the vendor serving processors (SGLang and vLLM agree):
//! smart-resize to a multiple of patch*merge, torch bilinear resampling
//! (align_corners=False, no antialias) on 0..255 floats, then ImageNet mean/std
//! scaled by 255. The patch layout is Qwen2-VL's (`qwen_vision.buildPixelValues`).

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const qwen_vision = @import("qwen_vision.zig");
const log = @import("log.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

pub const Resized = qwen_vision.Resized;

const PIXEL_MEAN = [3]f32{ 123.675, 116.28, 103.53 };
const PIXEL_STD = [3]f32{ 58.395, 57.12, 57.375 };

/// The vendor `smart_resize`: an image under `factor` on a side is first scaled
/// up to it; then each side snaps to a multiple of `factor` (Python `round`),
/// rescaled into [min_pixels, max_pixels] keeping the aspect ratio.
pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u32, max_pixels: u32) Resized {
    var fh: f64 = @floatFromInt(height);
    var fw: f64 = @floatFromInt(width);
    const ff: f64 = @floatFromInt(factor);
    const fmin: f64 = @floatFromInt(min_pixels);
    const fmax: f64 = @floatFromInt(max_pixels);
    if (@min(fh, fw) < ff) {
        const scale = ff / @min(fh, fw);
        fh = qwen_vision.roundHalfEven(fh * scale);
        fw = qwen_vision.roundHalfEven(fw * scale);
    }
    var h_bar = qwen_vision.roundHalfEven(fh / ff) * ff;
    var w_bar = qwen_vision.roundHalfEven(fw / ff) * ff;
    if (h_bar * w_bar > fmax) {
        const beta = @sqrt((fh * fw) / fmax);
        h_bar = @max(ff, std.math.floor(fh / beta / ff) * ff);
        w_bar = @max(ff, std.math.floor(fw / beta / ff) * ff);
    } else if (h_bar * w_bar < fmin) {
        const beta = @sqrt(fmin / (fh * fw));
        h_bar = std.math.ceil(fh * beta / ff) * ff;
        w_bar = std.math.ceil(fw * beta / ff) * ff;
    }
    return .{ .h = @intFromFloat(h_bar), .w = @intFromFloat(w_bar) };
}

/// One output coordinate of torch's bilinear source index (align_corners=False):
/// the two source taps and the far tap's weight, in f32 as torch computes them.
const Tap = struct { lo: u32, hi: u32, l1: f32 };

fn bilinearTap(dst: u32, in_len: u32, out_len: u32) Tap {
    const scale: f32 = @as(f32, @floatFromInt(in_len)) / @as(f32, @floatFromInt(out_len));
    const src = @max(scale * (@as(f32, @floatFromInt(dst)) + 0.5) - 0.5, 0);
    const lo: u32 = @intFromFloat(src);
    return .{ .lo = lo, .hi = if (lo + 1 < in_len) lo + 1 else lo, .l1 = src - @as(f32, @floatFromInt(lo)) };
}

/// Interleaved RGB8 [sh, sw, 3] -> normalized f32 CHW [3, dh, dw].
pub fn resizeNormalizedChw(dst: []f32, rgb: []const u8, sh: u32, sw: u32, dh: u32, dw: u32) !void {
    if (sh == 0 or sw == 0 or dh == 0 or dw == 0) return error.InvalidImageDimensions;
    const plane: usize = @as(usize, dh) * dw;
    if (dst.len != 3 * plane or rgb.len != @as(usize, sh) * sw * 3) return error.InvalidImageDimensions;
    for (0..dh) |y| {
        const ty = bilinearTap(@intCast(y), sh, dh);
        for (0..dw) |x| {
            const tx = bilinearTap(@intCast(x), sw, dw);
            for (0..3) |c| {
                const a = pixel(rgb, sw, ty.lo, tx.lo, c);
                const b = pixel(rgb, sw, ty.lo, tx.hi, c);
                const d = pixel(rgb, sw, ty.hi, tx.lo, c);
                const e = pixel(rgb, sw, ty.hi, tx.hi, c);
                const top = (1 - tx.l1) * a + tx.l1 * b;
                const bottom = (1 - tx.l1) * d + tx.l1 * e;
                const v = (1 - ty.l1) * top + ty.l1 * bottom;
                dst[c * plane + y * dw + x] = (v - PIXEL_MEAN[c]) / PIXEL_STD[c];
            }
        }
    }
}

fn pixel(rgb: []const u8, sw: u32, y: u32, x: u32, c: usize) f32 {
    return @floatFromInt(rgb[(@as(usize, y) * sw + x) * 3 + c]);
}

// ── The tower ──
//
// Port of `MiMoVisionTransformer` (modeling_mimo_v2.py) for one still image:
// Conv3d patch embed, 2-D rotary positions, RMSNorm blocks with GQA attention
// and a SwiGLU MLP, then LayerNorm + a two-layer GELU merger over each 2x2
// merge unit. Band blocks attend |i - j| <= window over the image's patches,
// in row-major or column-major merge-unit order, and add the block's per-head
// sink to key 0's logit (the reference's reading; vLLM's too).

const Block = struct {
    attn: model_mod.MimoVitAttn,
    norm1: mlx.mlx_array,
    norm2: mlx.mlx_array,
    qkv: Lin,
    proj: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
    sinks: ?mlx.mlx_array,
};

const Lin = struct { w: mlx.mlx_array, b: ?mlx.mlx_array = null };

pub const MimoVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    merge: u32,
    window: u32,
    out_hidden: u32,
    /// The Conv3d patch weight as its [hidden, C*T*P*P] Linear; owned when
    /// init had to flatten the stored rank-5 tensor.
    patch_w: mlx.mlx_array,
    patch_w_owned: bool,
    blocks: []Block,
    ln_q: mlx.mlx_array,
    fc1: Lin,
    fc2: Lin,

    const EPS: f32 = 1e-6;
    const ROPE_THETA: f32 = 10000.0;

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !MimoVision {
        var name_buf: [128]u8 = undefined;
        const s = mlx.gpuStream();
        const stored_patch = try must(weights, &name_buf, "visual.patch_embed.proj.weight", .{});
        var patch_w = stored_patch;
        var owned = false;
        if (mlx.getShape(stored_patch).len != 2) {
            const shape = mlx.getShape(stored_patch);
            var cols: c_int = 1;
            for (shape[1..]) |d| cols *= d;
            const flat = [_]c_int{ shape[0], cols };
            patch_w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&patch_w, stored_patch, &flat, 2, s));
            owned = true;
        }
        errdefer if (owned) {
            _ = mlx.mlx_array_free(patch_w);
        };

        const blocks = try allocator.alloc(Block, config.qv_depth);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*blk, i| {
            const band = config.mvit_attn[i] != .full;
            blk.* = .{
                .attn = config.mvit_attn[i],
                .norm1 = try must(weights, &name_buf, "visual.blocks.{d}.norm1.weight", .{i}),
                .norm2 = try must(weights, &name_buf, "visual.blocks.{d}.norm2.weight", .{i}),
                .qkv = try lin(weights, &name_buf, "visual.blocks.{d}.attn.qkv", .{i}),
                .proj = try lin(weights, &name_buf, "visual.blocks.{d}.attn.proj", .{i}),
                .gate = try lin(weights, &name_buf, "visual.blocks.{d}.mlp.gate_proj", .{i}),
                .up = try lin(weights, &name_buf, "visual.blocks.{d}.mlp.up_proj", .{i}),
                .down = try lin(weights, &name_buf, "visual.blocks.{d}.mlp.down_proj", .{i}),
                .sinks = if (band and config.mvit_sinks) try must(weights, &name_buf, "visual.blocks.{d}.attn.sinks", .{i}) else null,
            };
        }
        log.info("Vision encoder: MiMo-ViT (depth={d}, hidden={d}, heads={d}/{d}, window={d}, merge={d}, out_hidden={d})\n", .{
            config.qv_depth, config.qv_hidden, config.qv_heads, config.mvit_kv_heads, config.mvit_window, config.qv_merge, config.qv_out_hidden,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .hidden = config.qv_hidden,
            .heads = config.qv_heads,
            .kv_heads = config.mvit_kv_heads,
            .head_dim = config.qv_head_dim,
            .merge = config.qv_merge,
            .window = config.mvit_window,
            .out_hidden = config.qv_out_hidden,
            .patch_w = patch_w,
            .patch_w_owned = owned,
            .blocks = blocks,
            .ln_q = try must(weights, &name_buf, "visual.merger.ln_q.weight", .{}),
            .fc1 = try lin(weights, &name_buf, "visual.merger.mlp.0", .{}),
            .fc2 = try lin(weights, &name_buf, "visual.merger.mlp.2", .{}),
        };
    }

    pub fn deinit(self: *MimoVision) void {
        if (self.patch_w_owned) _ = mlx.mlx_array_free(self.patch_w);
        self.allocator.free(self.blocks);
    }

    /// `patches` is the processor's pixel_values [N, C*T*P*P] in merge-block
    /// order; returns [1, N/merge², out_hidden] bf16. The residual stream,
    /// norms and attention run in f32 around bf16 matmuls: an all-bf16 tower
    /// (ours, or the reference's own) drifts to cos 0.997 by the last block.
    pub fn forward(self: *MimoVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const a = self.allocator;
        const n: usize = @as(usize, grid_h) * grid_w;
        const unit: usize = @as(usize, self.merge) * self.merge;
        if (n == 0 or grid_h % self.merge != 0 or grid_w % self.merge != 0) return error.InvalidPatchGrid;

        var x = try astype(patches, .float32, self.s);
        defer _ = mlx.mlx_array_free(x);
        replace(&x, try self.linear(x, .{ .w = self.patch_w }));

        // Row-major positions, then their column-major permutation of whole merge units.
        const col_order = try colUnitOrder(a, grid_h / self.merge, grid_w / self.merge, unit);
        defer a.free(col_order);
        const inverse = try a.alloc(i32, n);
        defer a.free(inverse);
        for (col_order, 0..) |src, dst| inverse[@intCast(src)] = @intCast(dst);
        const rope_row = try self.buildRope(grid_w, n, null);
        defer rope_row.deinit();
        const rope_col = try self.buildRope(grid_w, n, col_order);
        defer rope_col.deinit();
        const masks = try BandMasks.init(a, n, self.window, self.s);
        defer masks.deinit();

        var in_col = false;
        for (self.blocks) |blk| {
            const want_col = blk.attn == .col;
            if (want_col != in_col) {
                replace(&x, try self.gather(x, if (want_col) col_order else inverse));
                in_col = want_col;
            }
            const normed = try rmsNorm(x, blk.norm1, self.s);
            defer _ = mlx.mlx_array_free(normed);
            const attn = try self.attention(normed, blk, if (in_col) rope_col else rope_row, masks, n);
            defer _ = mlx.mlx_array_free(attn);
            replace(&x, try binary(mlx.mlx_add, x, attn, self.s));

            const normed2 = try rmsNorm(x, blk.norm2, self.s);
            defer _ = mlx.mlx_array_free(normed2);
            const mlp = try self.swiglu(normed2, blk);
            defer _ = mlx.mlx_array_free(mlp);
            replace(&x, try binary(mlx.mlx_add, x, mlp, self.s));
            // One block's buffers at a time: that is what `encodeScratchBytes` bills.
            try mlx.check(mlx.mlx_array_eval(x));
        }
        // The reference hands the merger whatever order the last block used.

        {
            const wf = try astype(self.ln_q, .float32, self.s);
            defer _ = mlx.mlx_array_free(wf);
            var normed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_layer_norm(&normed, x, wf, .{ .ctx = null }, EPS, self.s));
            replace(&x, normed);
        }
        const merged: c_int = @intCast(n / unit);
        const wide = [_]c_int{ merged, @intCast(unit * self.hidden) };
        replace(&x, try reshape(x, &wide, self.s));
        replace(&x, try self.linear(x, self.fc1));
        replace(&x, try gelu(x, self.s));
        replace(&x, try self.linear(x, self.fc2));
        replace(&x, try astype(x, .bfloat16, self.s));
        // Encode now: images queued lazily would hold their scratch together.
        try mlx.check(mlx.mlx_array_eval(x));
        const out_shape = [_]c_int{ 1, merged, @intCast(self.out_hidden) };
        return reshape(x, &out_shape, self.s);
    }

    fn attention(self: *MimoVision, x: mlx.mlx_array, blk: Block, rope: Rope, masks: BandMasks, n: usize) !mlx.mlx_array {
        const h: c_int = @intCast(self.heads);
        const kvh: c_int = @intCast(self.kv_heads);
        const d: c_int = @intCast(self.head_dim);
        const nn: c_int = @intCast(n);
        const qkv = try self.linear(x, blk.qkv);
        defer _ = mlx.mlx_array_free(qkv);
        const q_end = h * d;
        const k_end = q_end + kvh * d;
        const q = try self.headsView(qkv, 0, q_end, h, nn, rope);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.headsView(qkv, q_end, k_end, kvh, nn, rope);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.headsView(qkv, k_end, k_end + kvh * d, kvh, nn, null);
        defer _ = mlx.mlx_array_free(v);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const ctx = if (blk.attn == .full)
            try sdpa(q, k, v, scale, null, self.s)
        else
            try self.bandAttention(q, k, v, scale, blk.sinks, masks, n);
        defer _ = mlx.mlx_array_free(ctx);
        // [1, H, N, D] -> [N, H*D]
        const t = try transposeAxes(ctx, &.{ 0, 2, 1, 3 }, self.s);
        defer _ = mlx.mlx_array_free(t);
        const flat = [_]c_int{ nn, h * d };
        const rows = try reshape(t, &flat, self.s);
        defer _ = mlx.mlx_array_free(rows);
        return self.linear(rows, blk.proj);
    }

    /// Columns [lo, hi) of the fused qkv as [1, heads, N, D], roped when asked.
    fn headsView(self: *MimoVision, qkv: mlx.mlx_array, lo: c_int, hi: c_int, heads: c_int, n: c_int, rope: ?Rope) !mlx.mlx_array {
        const d: c_int = @intCast(self.head_dim);
        const part = try slice(qkv, &.{ 0, lo }, &.{ n, hi }, self.s);
        defer _ = mlx.mlx_array_free(part);
        const shape = [_]c_int{ n, heads, d };
        var r = try reshape(part, &shape, self.s);
        defer _ = mlx.mlx_array_free(r);
        if (rope) |rp| replace(&r, try self.applyRope(r, rp));
        const t = try transposeAxes(r, &.{ 1, 0, 2 }, self.s);
        defer _ = mlx.mlx_array_free(t);
        const out_shape = [_]c_int{ 1, heads, n, d };
        return reshape(t, &out_shape, self.s);
    }

    /// Band attention in query blocks of `window` rows against the 3*window
    /// keys around them. Only the first two blocks can see key 0, so only they
    /// carry the per-head sink bias.
    fn bandAttention(self: *MimoVision, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, sinks: ?mlx.mlx_array, masks: BandMasks, n: usize) !mlx.mlx_array {
        const h: c_int = @intCast(self.heads);
        const d: c_int = @intCast(self.head_dim);
        const nb: c_int = @intCast(masks.blocks);
        const b: c_int = @intCast(self.window);
        const span: c_int = 3 * b;
        const padded: c_int = nb * b;

        const qp = try padAxis(q, 2, 0, padded - @as(c_int, @intCast(n)), self.s);
        defer _ = mlx.mlx_array_free(qp);
        const qb = try blocksOf(qp, h, nb, b, d, self.s);
        defer _ = mlx.mlx_array_free(qb);
        const kb = try self.keyWindows(k, masks, n);
        defer _ = mlx.mlx_array_free(kb);
        const vb = try self.keyWindows(v, masks, n);
        defer _ = mlx.mlx_array_free(vb);

        const head: c_int = @min(nb, 2);
        var parts: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } };
        defer for (parts) |p| if (p.ctx != null) {
            _ = mlx.mlx_array_free(p);
        };
        {
            // The first blocks: the shared band mask plus sink[h] on key 0's column.
            const base = try slice(masks.mask, &.{ 0, 0, 0, 0 }, &.{ head, 1, b, span }, self.s);
            defer _ = mlx.mlx_array_free(base);
            var mask = base;
            var owned = false;
            defer if (owned) {
                _ = mlx.mlx_array_free(mask);
            };
            if (sinks) |sk| {
                const sink_hd = [_]c_int{ 1, h, 1, 1 };
                const s4 = try reshape(sk, &sink_hd, self.s);
                defer _ = mlx.mlx_array_free(s4);
                const key0 = try slice(masks.key0, &.{ 0, 0, 0, 0 }, &.{ head, 1, 1, span }, self.s);
                defer _ = mlx.mlx_array_free(key0);
                const bias = try binary(mlx.mlx_multiply, key0, s4, self.s);
                defer _ = mlx.mlx_array_free(bias);
                mask = try binary(mlx.mlx_add, base, bias, self.s);
                owned = true;
            }
            const q0 = try slice(qb, &.{ 0, 0, 0, 0 }, &.{ head, h, b, d }, self.s);
            defer _ = mlx.mlx_array_free(q0);
            const k0 = try slice(kb, &.{ 0, 0, 0, 0 }, &.{ head, @intCast(self.kv_heads), span, d }, self.s);
            defer _ = mlx.mlx_array_free(k0);
            const v0 = try slice(vb, &.{ 0, 0, 0, 0 }, &.{ head, @intCast(self.kv_heads), span, d }, self.s);
            defer _ = mlx.mlx_array_free(v0);
            parts[0] = try sdpa(q0, k0, v0, scale, mask, self.s);
        }
        if (nb > head) {
            const kvh: c_int = @intCast(self.kv_heads);
            const q1 = try slice(qb, &.{ head, 0, 0, 0 }, &.{ nb, h, b, d }, self.s);
            defer _ = mlx.mlx_array_free(q1);
            const k1 = try slice(kb, &.{ head, 0, 0, 0 }, &.{ nb, kvh, span, d }, self.s);
            defer _ = mlx.mlx_array_free(k1);
            const v1 = try slice(vb, &.{ head, 0, 0, 0 }, &.{ nb, kvh, span, d }, self.s);
            defer _ = mlx.mlx_array_free(v1);
            const m1 = try slice(masks.mask, &.{ head, 0, 0, 0 }, &.{ nb, 1, b, span }, self.s);
            defer _ = mlx.mlx_array_free(m1);
            parts[1] = try sdpa(q1, k1, v1, scale, m1, self.s);
        }
        const out = if (parts[1].ctx != null) blk: {
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var joined = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, self.s));
            break :blk joined;
        } else try copyOf(parts[0], self.s);
        defer _ = mlx.mlx_array_free(out);
        // [nb, H, B, D] -> [1, H, nb*B, D] -> the first N rows.
        const t = try transposeAxes(out, &.{ 1, 0, 2, 3 }, self.s);
        defer _ = mlx.mlx_array_free(t);
        const rows = [_]c_int{ 1, h, padded, d };
        const r = try reshape(t, &rows, self.s);
        defer _ = mlx.mlx_array_free(r);
        return slice(r, &.{ 0, 0, 0, 0 }, &.{ 1, h, @intCast(n), d }, self.s);
    }

    /// [1, Hkv, N, D] -> [nb, Hkv, 3B, D]: each block's keys from row start-B.
    fn keyWindows(self: *MimoVision, k: mlx.mlx_array, masks: BandMasks, n: usize) !mlx.mlx_array {
        const b: c_int = @intCast(self.window);
        const nb: c_int = @intCast(masks.blocks);
        const kvh: c_int = @intCast(self.kv_heads);
        const d: c_int = @intCast(self.head_dim);
        const kp = try padAxis(k, 2, b, nb * b + b - @as(c_int, @intCast(n)), self.s);
        defer _ = mlx.mlx_array_free(kp);
        var g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&g, kp, masks.key_index, 2, self.s));
        defer _ = mlx.mlx_array_free(g);
        const shape = [_]c_int{ kvh, nb, 3 * b, d };
        const r = try reshape(g, &shape, self.s);
        defer _ = mlx.mlx_array_free(r);
        return transposeAxes(r, &.{ 1, 0, 2, 3 }, self.s);
    }

    fn swiglu(self: *MimoVision, x: mlx.mlx_array, blk: Block) !mlx.mlx_array {
        const g = try self.linear(x, blk.gate);
        defer _ = mlx.mlx_array_free(g);
        const sg = try unary(mlx.mlx_sigmoid, g, self.s);
        defer _ = mlx.mlx_array_free(sg);
        const act = try binary(mlx.mlx_multiply, g, sg, self.s);
        defer _ = mlx.mlx_array_free(act);
        const u = try self.linear(x, blk.up);
        defer _ = mlx.mlx_array_free(u);
        const prod = try binary(mlx.mlx_multiply, act, u, self.s);
        defer _ = mlx.mlx_array_free(prod);
        return self.linear(prod, blk.down);
    }

    /// bf16 matmul (and bias) on the f32 stream, widened back to f32.
    fn linear(self: *MimoVision, x: mlx.mlx_array, l: Lin) !mlx.mlx_array {
        const xb = try astype(x, .bfloat16, self.s);
        defer _ = mlx.mlx_array_free(xb);
        const wt = try unary(mlx.mlx_transpose, l.w, self.s);
        defer _ = mlx.mlx_array_free(wt);
        var y = try binary(mlx.mlx_matmul, xb, wt, self.s);
        defer _ = mlx.mlx_array_free(y);
        if (l.b) |bias| replace(&y, try binary(mlx.mlx_add, y, bias, self.s));
        return astype(y, .float32, self.s);
    }

    fn gather(self: *MimoVision, x: mlx.mlx_array, order: []const i32) !mlx.mlx_array {
        const idx = hostI32(order);
        defer _ = mlx.mlx_array_free(idx);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&out, x, idx, 0, self.s));
        return out;
    }

    const Rope = struct {
        cos: mlx.mlx_array,
        sin: mlx.mlx_array,

        fn deinit(self: Rope) void {
            _ = mlx.mlx_array_free(self.cos);
            _ = mlx.mlx_array_free(self.sin);
        }
    };

    /// cos/sin [N, 1, D] f32 for each patch in merge-block order (permuted by
    /// `order` when given): D/4 frequencies of the row, D/4 of the column,
    /// the pair repeated, as `rot_pos_emb` lays them out.
    fn buildRope(self: *MimoVision, grid_w: u32, count: usize, order: ?[]const i32) !Rope {
        const a = self.allocator;
        const d: usize = self.head_dim;
        const quarter = d / 4;
        const m = self.merge;
        const blocks_w = grid_w / m;
        const cosv = try a.alloc(f32, count * d);
        defer a.free(cosv);
        const sinv = try a.alloc(f32, count * d);
        defer a.free(sinv);
        for (0..count) |slot| {
            const p: usize = if (order) |o| @intCast(o[slot]) else slot;
            const unit_i = p / (m * m);
            const within = p % (m * m);
            const row: f32 = @floatFromInt((unit_i / blocks_w) * m + within / m);
            const col: f32 = @floatFromInt((unit_i % blocks_w) * m + within % m);
            for (0..quarter) |j| {
                const inv = 1.0 / std.math.pow(f32, ROPE_THETA, @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(2 * quarter)));
                for ([_]usize{ j, j + quarter }, [_]f32{ row, col }) |at, pos| {
                    const angle = pos * inv;
                    for ([_]usize{ at, at + 2 * quarter }) |k| {
                        cosv[slot * d + k] = @cos(angle);
                        sinv[slot * d + k] = @sin(angle);
                    }
                }
            }
        }
        const shape = [_]c_int{ @intCast(count), 1, @intCast(d) };
        return .{
            .cos = mlx.mlx_array_new_data(cosv.ptr, &shape, 3, .float32),
            .sin = mlx.mlx_array_new_data(sinv.ptr, &shape, 3, .float32),
        };
    }

    /// x·cos + rotate_half(x)·sin over [N, heads, D], in f32 like the reference.
    fn applyRope(self: *MimoVision, x: mlx.mlx_array, r: Rope) !mlx.mlx_array {
        const shape = mlx.getShape(x);
        const n = shape[0];
        const heads = shape[1];
        const d = shape[2];
        const half = @divExact(d, 2);
        const xf = try astype(x, .float32, self.s);
        defer _ = mlx.mlx_array_free(xf);
        const x1 = try slice(xf, &.{ 0, 0, 0 }, &.{ n, heads, half }, self.s);
        defer _ = mlx.mlx_array_free(x1);
        const x2 = try slice(xf, &.{ 0, 0, half }, &.{ n, heads, d }, self.s);
        defer _ = mlx.mlx_array_free(x2);
        const neg = try unary(mlx.mlx_negative, x2, self.s);
        defer _ = mlx.mlx_array_free(neg);
        const pair = [_]mlx.mlx_array{ neg, x1 };
        const vec = mlx.mlx_vector_array_new_data(&pair, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var rot = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&rot, vec, -1, self.s));
        defer _ = mlx.mlx_array_free(rot);
        const xc = try binary(mlx.mlx_multiply, xf, r.cos, self.s);
        defer _ = mlx.mlx_array_free(xc);
        const rs = try binary(mlx.mlx_multiply, rot, r.sin, self.s);
        defer _ = mlx.mlx_array_free(rs);
        return binary(mlx.mlx_add, xc, rs, self.s);
    }
};

/// Transient bytes one image's encode needs on top of the resident tower, at
/// `n_patches` patches: one block's f32 activations (the stream is evaluated
/// per block) plus a fixed floor. Covers the measured peak (`mimo vision
/// ubench`) with >= 25% margin from 196 to 9216 patches.
pub fn encodeScratchBytes(config: *const ModelConfig, n_patches: u64) u64 {
    const per_patch: u64 = 4 * (8 * @as(u64, config.qv_hidden) + 6 * @as(u64, config.qv_intermediate) +
        6 * @as(u64, config.qv_heads + 2 * config.mvit_kv_heads) * config.qv_head_dim);
    return (64 << 20) + n_patches * per_patch;
}

/// The key rows and masks a band attention reuses across its blocks, built
/// once per image. Query block b holds rows [bB, bB+B) with B = window and
/// sees key rows [bB-B, bB+2B), zero-padded past both ends.
const BandMasks = struct {
    blocks: usize,
    /// [nb, 1, B, 3B] f32: 0 where |i - j| <= window and 0 <= j < n, else -inf.
    mask: mlx.mlx_array,
    /// [nb, 1, 1, 3B] f32: 1 at key 0's column.
    key0: mlx.mlx_array,
    /// [nb*3B] rows of the front-padded keys each block reads.
    key_index: mlx.mlx_array,

    fn init(a: std.mem.Allocator, n: usize, window: u32, s: mlx.mlx_stream) !BandMasks {
        const b: usize = window;
        const nb = (n + b - 1) / b;
        const span = 3 * b;
        const mask = try a.alloc(f32, nb * b * span);
        defer a.free(mask);
        const key0 = try a.alloc(f32, nb * span);
        defer a.free(key0);
        const index = try a.alloc(i32, nb * span);
        defer a.free(index);
        @memset(key0, 0);
        for (0..nb) |blk| {
            for (0..span) |c| {
                index[blk * span + c] = @intCast(blk * b + c);
                if (blk * b + c == b) key0[blk * span + c] = 1;
            }
            for (0..b) |r| {
                const i: isize = @intCast(blk * b + r);
                for (0..span) |c| {
                    const j: isize = @as(isize, @intCast(blk * b + c)) - @as(isize, @intCast(b));
                    const inside = j >= 0 and j < n and @abs(i - j) <= b;
                    mask[(blk * b + r) * span + c] = if (inside) 0 else -std.math.inf(f32);
                }
            }
        }
        const mask_shape = [_]c_int{ @intCast(nb), 1, @intCast(b), @intCast(span) };
        const key0_shape = [_]c_int{ @intCast(nb), 1, 1, @intCast(span) };
        const mask_f = mlx.mlx_array_new_data(mask.ptr, &mask_shape, 4, .float32);
        defer _ = mlx.mlx_array_free(mask_f);
        const key0_f = mlx.mlx_array_new_data(key0.ptr, &key0_shape, 4, .float32);
        defer _ = mlx.mlx_array_free(key0_f);
        return .{
            .blocks = nb,
            .mask = try copyOf(mask_f, s),
            .key0 = try copyOf(key0_f, s),
            .key_index = hostI32(index),
        };
    }

    fn deinit(self: BandMasks) void {
        _ = mlx.mlx_array_free(self.mask);
        _ = mlx.mlx_array_free(self.key0);
        _ = mlx.mlx_array_free(self.key_index);
    }
};

/// Patch order of the column-major blocks: whole merge units, column-major.
fn colUnitOrder(a: std.mem.Allocator, units_h: u32, units_w: u32, unit: usize) ![]i32 {
    const out = try a.alloc(i32, @as(usize, units_h) * units_w * unit);
    var at: usize = 0;
    for (0..units_w) |bw| for (0..units_h) |bh| {
        const u = bh * units_w + bw;
        for (0..unit) |r| {
            out[at] = @intCast(u * unit + r);
            at += 1;
        }
    };
    return out;
}

fn must(weights: *const Weights, buf: *[128]u8, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = std.fmt.bufPrint(buf, fmt, args) catch return error.NameTooLong;
    return weights.get(key) orelse {
        log.warn("MISSING MIMO VISION WEIGHT: {s}\n", .{key});
        return error.MissingVisionWeights;
    };
}

fn lin(weights: *const Weights, buf: *[128]u8, comptime base: []const u8, args: anytype) !Lin {
    const w = try must(weights, buf, base ++ ".weight", args);
    const key = std.fmt.bufPrint(buf, base ++ ".bias", args) catch return error.NameTooLong;
    return .{ .w = w, .b = weights.get(key) };
}

fn replace(dst: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(dst.*);
    dst.* = next;
}

fn hostI32(v: []const i32) mlx.mlx_array {
    const shape = [_]c_int{@intCast(v.len)};
    return mlx.mlx_array_new_data(v.ptr, &shape, 1, .int32);
}

fn unary(comptime f: anytype, a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(f(&out, a, s));
    return out;
}

fn binary(comptime f: anytype, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(f(&out, a, b, s));
    return out;
}

fn astype(a: mlx.mlx_array, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, a, dtype, s));
    return out;
}

fn copyOf(a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&out, a, false, s));
    return out;
}

fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const wf = try astype(w, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, wf, MimoVision.EPS, s));
    return out;
}

fn reshape(a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, a, shape.ptr, shape.len, s));
    return out;
}

fn transposeAxes(a: mlx.mlx_array, axes: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, a, axes.ptr, axes.len, s));
    return out;
}

fn slice(a: mlx.mlx_array, start: []const c_int, stop: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var strides: [4]c_int = @splat(1);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, start.ptr, start.len, stop.ptr, stop.len, &strides, start.len, s));
    return out;
}

/// Zero-pad `axis` by `low` rows in front and `high` behind.
fn padAxis(a: mlx.mlx_array, axis: c_int, low: c_int, high: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (low == 0 and high == 0) return copyOf(a, s);
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    const axes = [_]c_int{axis};
    const lo = [_]c_int{low};
    const hi = [_]c_int{high};
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(&out, a, &axes, 1, &lo, 1, &hi, 1, zero, "constant", s));
    return out;
}

/// [1, H, nb*B, D] -> [nb, H, B, D].
fn blocksOf(a: mlx.mlx_array, heads: c_int, nb: c_int, b: c_int, d: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = [_]c_int{ heads, nb, b, d };
    const r = try reshape(a, &shape, s);
    defer _ = mlx.mlx_array_free(r);
    return transposeAxes(r, &.{ 1, 0, 2, 3 }, s);
}

fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    const none = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, if (mask != null) "array" else "", mask orelse none, none, false, s));
    return out;
}

/// nn.GELU (exact erf form) on the f32 stream.
fn gelu(xf: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const t = try binary(mlx.mlx_multiply, xf, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(t);
    const e = try unary(mlx.mlx_erf, t, s);
    defer _ = mlx.mlx_array_free(e);
    const onep = try binary(mlx.mlx_add, e, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const xh = try binary(mlx.mlx_multiply, xf, half, s);
    defer _ = mlx.mlx_array_free(xh);
    return binary(mlx.mlx_multiply, xh, onep, s);
}

fn sumProduct(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const af = try astype(a, .float32, s);
    defer _ = mlx.mlx_array_free(af);
    const bf = try astype(b, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const flat = [_]c_int{-1};
    const a1 = try reshape(af, &flat, s);
    defer _ = mlx.mlx_array_free(a1);
    const b1 = try reshape(bf, &flat, s);
    defer _ = mlx.mlx_array_free(b1);
    const prod = try binary(mlx.mlx_multiply, a1, b1, s);
    defer _ = mlx.mlx_array_free(prod);
    var total = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(total);
    try mlx.check(mlx.mlx_sum(&total, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(total));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, total));
    return v;
}

/// Cosine of two tensors read as flat vectors, in f32. NaN fails every bar.
pub fn cosineSim(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const ab = try sumProduct(a, b, s);
    const aa = try sumProduct(a, a, s);
    const bb = try sumProduct(b, b, s);
    if (!std.math.isFinite(ab) or aa <= 0 or bb <= 0) return std.math.nan(f32);
    return ab / (@sqrt(aa) * @sqrt(bb));
}

/// ||a|| / ||b||: the scale error a cosine cannot see.
pub fn rmsRatio(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const aa = try sumProduct(a, a, s);
    const bb = try sumProduct(b, b, s);
    if (!std.math.isFinite(aa) or bb <= 0) return std.math.nan(f32);
    return @sqrt(aa / bb);
}

// ── Tests ──

const testing = std.testing;

test "smartResize matches the vendor processor, tiny sides and the engine ceiling included" {
    // Expected values from the vendor `smart_resize` (factor 32, min 8192, max 1536²).
    const cases = [_]struct { in: [2]u32, out: [2]u32 }{
        .{ .in = .{ 1080, 1920 }, .out = .{ 1088, 1920 } },
        .{ .in = .{ 1920, 1080 }, .out = .{ 1920, 1088 } },
        .{ .in = .{ 7, 300 }, .out = .{ 32, 1376 } },
        .{ .in = .{ 50, 50 }, .out = .{ 96, 96 } },
        .{ .in = .{ 3000, 4000 }, .out = .{ 1312, 1760 } },
        .{ .in = .{ 112, 112 }, .out = .{ 128, 128 } },
        .{ .in = .{ 48, 80 }, .out = .{ 96, 128 } },
        .{ .in = .{ 33, 33 }, .out = .{ 96, 96 } },
        .{ .in = .{ 2000, 30 }, .out = .{ 2144, 32 } },
    };
    for (cases) |case| {
        const r = smartResize(case.in[0], case.in[1], 32, 8192, 1536 * 1536);
        try testing.expectEqual(case.out[0], r.h);
        try testing.expectEqual(case.out[1], r.w);
    }
}

test "resizeNormalizedChw matches torch bilinear (align_corners=False) and ImageNet mean/std x255" {
    const h = 5;
    const w = 7;
    var rgb: [h * w * 3]u8 = undefined;
    for (0..h) |r| for (0..w) |c| for (0..3) |ch| {
        rgb[(r * w + c) * 3 + ch] = @intCast((r * 37 + c * 91 + ch * 53) % 256);
    };
    // torch.nn.functional.interpolate on the float image, then (x - mean) / std.
    const want_3x4 = [_]f32{
        -1.3223165, 0.8568086,  -0.2520193, -0.2648630, -0.2662900, -1.9231099, 0.8040071,  -0.8528128, 0.7897363, -0.8670834, -0.6972623, 0.2032135,
        -0.2944969, 0.6260797,  0.7996909,  0.2263363,  0.7851016,  -0.9087010, 1.8792893,  0.1854867,  0.7442520, 0.1708975,  -0.9626810, 1.2650851,
        0.8527814,  -0.8334931, 1.9421062,  0.2558316,  0.2543791,  0.2413072,  -0.8872331, 1.3306319,  1.3291795, 0.9442846,  -0.1842557, 0.5463184,
    };
    var got: [3 * 3 * 4]f32 = undefined;
    try resizeNormalizedChw(&got, &rgb, h, w, 3, 4);
    for (want_3x4, got) |want, g| try testing.expectApproxEqAbs(want, g, 2e-5);

    // Upsampling: spot-check the corners and an interior sample of an 8x11 plane per channel.
    var up: [3 * 8 * 11]f32 = undefined;
    try resizeNormalizedChw(&up, &rgb, h, w, 8, 11);
    const picks = [_]struct { i: usize, v: f32 }{
        .{ .i = 0, .v = -2.1179039 },  .{ .i = 10, .v = -1.5356623 },  .{ .i = 87, .v = 0.9988012 },
        .{ .i = 88, .v = -1.1078432 }, .{ .i = 100, .v = -0.1002953 }, .{ .i = 175, .v = 2.0784314 },
        .{ .i = 176, .v = 0.0430501 }, .{ .i = 200, .v = 1.7993391 },  .{ .i = 263, .v = -1.2467102 },
    };
    for (picks) |p| try testing.expectApproxEqAbs(p.v, up[p.i], 2e-5);
}

fn tinyConfig() model_mod.ModelConfig {
    var c = model_mod.ModelConfig{};
    c.hidden_size = 64;
    c.mimo_vision = true;
    c.qv_depth = 4;
    c.qv_hidden = 32;
    c.qv_heads = 4;
    c.qv_head_dim = 16;
    c.mvit_kv_heads = 2;
    c.qv_intermediate = 48;
    c.qv_out_hidden = 64;
    c.qv_patch = 4;
    c.qv_merge = 2;
    c.qv_temporal_patch = 2;
    c.mvit_window = 4;
    c.mvit_sinks = true;
    c.mvit_attn[0] = .full;
    c.mvit_attn[1] = .row;
    c.mvit_attn[2] = .col;
    c.mvit_attn[3] = .full;
    return c;
}

/// The committed tiny fixture's weights as a Weights map (tests/dump_mimo_vision_fixtures.py tiny).
fn loadTinyFixture(tmp: *std.testing.TmpDir) !Weights {
    const io = testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "tiny.safetensors", .data = @embedFile("fixtures/mimo_vision_tiny.safetensors") });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(io, &buf);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/tiny.safetensors", .{buf[0..dir_len]});
    defer testing.allocator.free(path);
    return model_mod.loadWeightsSingleFile(testing.allocator, path);
}

test "mimo vision tiny: the tower matches the reference, with the sink as a bias on key 0's logit" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var weights = try loadTinyFixture(&tmp);
    defer weights.deinit();
    var tower = try MimoVision.init(testing.allocator, tinyConfig(), &weights);
    defer tower.deinit();

    const pv = weights.get("fixture.pixel_values") orelse return error.MissingFixtureTensor;
    const out = try tower.forward(pv, 8, 12);
    defer _ = mlx.mlx_array_free(out);
    try testing.expectEqualSlices(c_int, &.{ 1, 24, 64 }, mlx.getShape(out));

    const want = weights.get("fixture.features") orelse return error.MissingFixtureTensor;
    const other = weights.get("fixture.features_sink_column") orelse return error.MissingFixtureTensor;
    const cos = try cosineSim(out, want, tower.s);
    const ratio = try rmsRatio(out, want, tower.s);
    const cos_other = try cosineSim(out, other, tower.s);
    if (@import("transformer.zig").diagEnvOn("SUSHI_MIMO_VISION_DIAG"))
        std.debug.print("[mimo-vit tiny] cos={d:.6} rms_ratio={d:.5} cos_sink_column={d:.5}\n", .{ cos, ratio, cos_other });
    try testing.expect(cos > 0.9995);
    try testing.expect(ratio > 0.99 and ratio < 1.01);
    // The fixture's two sink readings differ (cos 0.98): ours must be the key-0 bias.
    try testing.expect(cos_other < 0.99);
}

test "VisionEncoder serves a MiMo config through the MiMo-ViT" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var weights = try loadTinyFixture(&tmp);
    defer weights.deinit();
    var enc = try @import("vision.zig").VisionEncoder.init(testing.allocator, tinyConfig(), &weights);
    defer enc.deinit();
    try testing.expect(enc.mimo != null and !enc.supportsAudio());
    const pv = weights.get("fixture.pixel_values") orelse return error.MissingFixtureTensor;
    const out = try enc.forwardPatches(pv, 8, 12);
    defer _ = mlx.mlx_array_free(out);
    try testing.expectEqualSlices(c_int, &.{ 1, 24, 64 }, mlx.getShape(out));
}

// Real-weight parity against the reference run on the CPU in f32
// (tests/dump_mimo_vision_fixtures.py real):
//   MIMO_V2_SOURCE=<checkpoint> MIMO_VISION_ORACLE=<real.safetensors> \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="mimo vision real"
test "mimo vision real: preprocessing and the bf16 tower match the f32 reference" {
    const raw_dir = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("MIMO_VISION_ORACLE") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    const dir = std.mem.span(raw_dir);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    var weights = Weights.init(a);
    defer weights.deinit();
    try @import("mimo_source.zig").loadVisionWeightsInto(&weights, io, a, dir);
    var fx = try model_mod.loadWeightsSingleFile(a, std.mem.span(raw_fix));
    defer fx.deinit();
    var tower = try MimoVision.init(a, config, &weights);
    defer tower.deinit();
    const s = tower.s;
    const diag = @import("transformer.zig").diagEnvOn("SUSHI_MIMO_VISION_DIAG");

    const grid = fx.get("grid_thw") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(grid));
    const g = mlx.mlx_array_data_int32(grid) orelse return error.MissingFixtureTensor;
    const gh: u32 = @intCast(g[1]);
    const gw: u32 = @intCast(g[2]);
    const want_pv = fx.get("pixel_values") orelse return error.MissingFixtureTensor;

    // Our preprocessing of the same RGB lands on the reference's patch grid and values.
    {
        const rgb_arr = fx.get("rgb") orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(rgb_arr));
        const shape = mlx.getShape(rgb_arr);
        const sh: u32 = @intCast(shape[0]);
        const sw: u32 = @intCast(shape[1]);
        const rgb = (mlx.mlx_array_data_uint8(rgb_arr) orelse return error.MissingFixtureTensor)[0 .. @as(usize, sh) * sw * 3];
        const bounds = qwen_vision.effectivePixelBounds(config.qv_min_pixels, config.qv_max_pixels);
        const rs = smartResize(sh, sw, config.qv_patch * config.qv_merge, bounds.min, bounds.max);
        try testing.expectEqual(gh * config.qv_patch, rs.h);
        try testing.expectEqual(gw * config.qv_patch, rs.w);
        const chw = try a.alloc(f32, 3 * @as(usize, rs.h) * rs.w);
        defer a.free(chw);
        try resizeNormalizedChw(chw, rgb, sh, sw, rs.h, rs.w);
        const feat: usize = 3 * config.qv_temporal_patch * config.qv_patch * config.qv_patch;
        const pv = try a.alloc(f32, @as(usize, gh) * gw * feat);
        defer a.free(pv);
        qwen_vision.buildPixelValues(pv, chw, 3, rs.h, rs.w, config.qv_patch, config.qv_temporal_patch, config.qv_merge);
        try mlx.check(mlx.mlx_array_eval(want_pv));
        const want = (mlx.mlx_array_data_float32(want_pv) orelse return error.MissingFixtureTensor)[0..pv.len];
        var worst: f32 = 0;
        for (pv, want) |x, y| worst = @max(worst, @abs(x - y));
        if (diag) std.debug.print("[mimo-vit real] pixel_values max|diff|={e}\n", .{worst});
        try testing.expect(worst < 1e-4);
    }

    const out = try tower.forward(want_pv, gh, gw);
    defer _ = mlx.mlx_array_free(out);
    const want = fx.get("features") orelse return error.MissingFixtureTensor;
    const cos = try cosineSim(out, want, s);
    const ratio = try rmsRatio(out, want, s);
    if (diag) std.debug.print("[mimo-vit real] grid {d}x{d} features cos={d:.6} rms_ratio={d:.5}\n", .{ gh, gw, cos, ratio });
    try testing.expect(cos > 0.999);
    try testing.expect(ratio > 0.99 and ratio < 1.01);
}

// Encode time and scratch peak on the real tower at a small image and at the
// engine's 1536² cap (random pixels; the numerics are the parity test's job):
//   MIMO_V2_SOURCE=<checkpoint> SUSHI_MIMO_VISION_UBENCH=<reps> \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="mimo vision ubench"
test "mimo vision ubench: encode time and scratch peak per image" {
    const reps_raw = std.c.getenv("SUSHI_MIMO_VISION_UBENCH") orelse return error.SkipZigTest;
    const raw_dir = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const reps = std.fmt.parseInt(u32, std.mem.span(reps_raw), 10) catch return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    const dir = std.mem.span(raw_dir);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    var weights = Weights.init(a);
    defer weights.deinit();
    try @import("mimo_source.zig").loadVisionWeightsInto(&weights, io, a, dir);
    var tower = try MimoVision.init(a, config, &weights);
    defer tower.deinit();
    const feat: c_int = @intCast(3 * config.qv_temporal_patch * config.qv_patch * config.qv_patch);
    var under_billed = false;
    for ([_][2]u32{ .{ 14, 14 }, .{ 68, 120 }, .{ 96, 96 } }) |g| {
        const n: c_int = @intCast(g[0] * g[1]);
        var pv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pv);
        const shape = [_]c_int{ n, feat };
        try mlx.check(mlx.mlx_random_normal(&pv, &shape, 2, .float32, 0, 1, .{ .ctx = null }, tower.s));
        try mlx.check(mlx.mlx_array_eval(pv));
        var best: u64 = std.math.maxInt(u64);
        var first_peak: usize = 0;
        var steady_peak: usize = 0;
        for (0..reps + 1) |r| {
            try mlx.check(mlx.mlx_synchronize(tower.s));
            var before: usize = 0;
            _ = mlx.mlx_get_active_memory(&before);
            _ = mlx.mlx_reset_peak_memory();
            const t0 = std.Io.Timestamp.now(io, .boot);
            const out = try tower.forward(pv, g[0], g[1]);
            try mlx.check(mlx.mlx_array_eval(out));
            try mlx.check(mlx.mlx_synchronize(tower.s));
            const ns: u64 = @intCast(t0.untilNow(io, .boot).nanoseconds);
            var peak: usize = 0;
            _ = mlx.mlx_get_peak_memory(&peak);
            _ = mlx.mlx_array_free(out);
            if (r == 0) first_peak = peak -| before else {
                best = @min(best, ns);
                steady_peak = @max(steady_peak, peak -| before);
            }
        }
        const bill = encodeScratchBytes(&config, @intCast(n));
        std.debug.print("[mimo-vit ubench] grid {d}x{d} ({d} patches, {d} tokens): best {d:.1} ms of {d}, scratch peak {d:.1} MB (first run {d:.1} MB), bill {d:.1} MB\n", .{
            g[0],                                       g[1],                                      n,                                   @divExact(n, 4), @as(f64, @floatFromInt(best)) / 1e6, reps,
            @as(f64, @floatFromInt(steady_peak)) / 1e6, @as(f64, @floatFromInt(first_peak)) / 1e6, @as(f64, @floatFromInt(bill)) / 1e6,
        });
        if (bill < @max(steady_peak, first_peak)) under_billed = true;
    }
    try testing.expect(!under_billed);
}
