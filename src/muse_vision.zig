//! Muse-Glimmer vision tower + preprocessing math.
//!
//! Port of `transformers/models/muse_glimmer/` (`modeling_muse_glimmer.py`
//! vision half, `image_processing_muse_glimmer.py`). Qwen2.5-VL-shaped ViT with
//! its own quirks: a plain-Linear patch embedder (no conv layout), a learned
//! 32x32 position table bilinearly resampled per image, interleaved
//! `[freq_w, freq_h, freq_w, freq_h]` 2D RoPE, and window/full attention
//! alternating per `layer_types`. Images only — video is not wired.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const log = @import("log.zig");

pub const Resized = qwen_vision.Resized;

/// `smart_resize`: pick the integer patch grid closest to the input aspect ratio
/// under `max_tokens`. `factor` is patch_size x merge_size, so the cap counts
/// MERGED tokens. Returns the resize target in pixels. The reference breaks
/// exact ties by CPython set-iteration order; we take the larger grid, which is
/// what it lands on for the square case where ties actually occur.
pub fn smartResize(height: u32, width: u32, factor: u32, max_tokens: u32) Resized {
    const fh: f64 = @floatFromInt(height);
    const fw: f64 = @floatFromInt(width);
    const ff: f64 = @floatFromInt(factor);
    const cap: f64 = @floatFromInt(max_tokens);

    var ideal_h = fh / ff;
    var ideal_w = fw / ff;
    const ratio = if (ideal_h > 0) ideal_w / ideal_h else 1.0;
    if (ideal_h * ideal_w > cap) {
        ideal_h = @sqrt(cap / ratio);
        ideal_w = ideal_h * ratio;
    }
    const target = fh / fw;

    var best_h: u32 = 1;
    var best_w: u32 = 1;
    var best_err: f64 = std.math.inf(f64);
    for ([_]f64{ @ceil(ideal_h), @floor(ideal_h) }) |ch| {
        for ([_]f64{ @ceil(ideal_w), @floor(ideal_w) }) |cw| {
            if (ch < 1 or cw < 1 or ch * cw > cap) continue;
            const err = @abs(ch / cw - target);
            if (err < best_err) {
                best_err = err;
                best_h = @intFromFloat(ch);
                best_w = @intFromFloat(cw);
            }
        }
    }
    return .{ .h = best_h * factor, .w = best_w * factor };
}

/// Build the processor's `pixel_values` [gh*gw, tps*C*ps*ps] from a normalized
/// CHW image. Row-major grid order (NOT Qwen's merge-block order — muse merges
/// later, in `pixel_shuffle`) and feature layout [t, c, py, px], the temporal
/// axis duplicating the single frame.
pub fn buildPixelValues(out: []f32, img_chw: []const f32, C: u32, rh: u32, rw: u32, patch: u32, tps: u32) void {
    const gh = rh / patch;
    const gw = rw / patch;
    const feat = tps * C * patch * patch;
    std.debug.assert(out.len == @as(usize, gh) * gw * feat);
    const plane: usize = @as(usize, rh) * rw;

    for (0..gh) |row| {
        for (0..gw) |col| {
            const base = (row * gw + col) * feat;
            var f: usize = 0;
            for (0..tps) |_| {
                for (0..C) |c| {
                    for (0..patch) |py| {
                        const y = row * patch + py;
                        for (0..patch) |px| {
                            const x = col * patch + px;
                            out[base + f] = img_chw[c * plane + y * rw + x];
                            f += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Windowed-attention patch permutation (`get_vision_window_index` with
/// spatial_merge_size 1): patches are grouped into `win`×`win` windows, each
/// window contiguous. `order[k]` is the row-major patch index at permuted
/// position k; `seqlens` holds the per-window token counts (edge windows are
/// short, empty ones dropped). Caller owns both slices.
pub const Windows = struct {
    order: []i32,
    seqlens: []i32,

    pub fn deinit(self: *Windows, a: std.mem.Allocator) void {
        a.free(self.order);
        a.free(self.seqlens);
    }
};

pub fn windowIndex(a: std.mem.Allocator, gh: u32, gw: u32, win: u32) !Windows {
    const n: usize = @as(usize, gh) * gw;
    var order = try std.ArrayList(i32).initCapacity(a, n);
    errdefer order.deinit(a);
    var seqlens = std.ArrayList(i32).empty;
    errdefer seqlens.deinit(a);

    const nwh = (gh + win - 1) / win;
    const nww = (gw + win - 1) / win;
    for (0..nwh) |wh| {
        for (0..nww) |ww| {
            var count: i32 = 0;
            for (0..win) |i| {
                const r = wh * win + i;
                if (r >= gh) continue;
                for (0..win) |j| {
                    const c = ww * win + j;
                    if (c >= gw) continue;
                    order.appendAssumeCapacity(@intCast(r * gw + c));
                    count += 1;
                }
            }
            if (count > 0) try seqlens.append(a, count);
        }
    }
    return .{ .order = try order.toOwnedSlice(a), .seqlens = try seqlens.toOwnedSlice(a) };
}

/// Bilinear resampling plan for one axis of the learned `side`×`side` position
/// table onto a `len`-patch grid. Mirrors `F.grid_sample(align_corners=False,
/// padding="zeros")`: out-of-range corners keep a clamped index but ZERO weight.
const Axis = struct {
    lo: []i32,
    hi: []i32,
    frac: []f64,
    lo_valid: []bool,
    hi_valid: []bool,

    fn deinit(self: *Axis, a: std.mem.Allocator) void {
        a.free(self.lo);
        a.free(self.hi);
        a.free(self.frac);
        a.free(self.lo_valid);
        a.free(self.hi_valid);
    }
};

fn bilinearAxis(a: std.mem.Allocator, len: u32, side: u32) !Axis {
    const lo = try a.alloc(i32, len);
    const hi = try a.alloc(i32, len);
    const frac = try a.alloc(f64, len);
    const lo_valid = try a.alloc(bool, len);
    const hi_valid = try a.alloc(bool, len);
    const last: i32 = @intCast(side - 1);
    const scale = @as(f64, @floatFromInt(side)) / @as(f64, @floatFromInt(len));
    for (0..len) |i| {
        const g = (@as(f64, @floatFromInt(i)) + 0.5) * scale - 0.5;
        const fl = @floor(g);
        const fi: i32 = @intFromFloat(fl);
        frac[i] = g - fl;
        lo_valid[i] = fi >= 0 and fi <= last;
        hi_valid[i] = fi + 1 >= 0 and fi + 1 <= last;
        lo[i] = std.math.clamp(fi, 0, last);
        hi[i] = std.math.clamp(fi + 1, 0, last);
    }
    return .{ .lo = lo, .hi = hi, .frac = frac, .lo_valid = lo_valid, .hi_valid = hi_valid };
}

// ─────────────────────────────────────────────────────────────────────────────
// ViT encoder. Single still image (grid_thw = [[1, gh, gw]]).
// forward(pixel_values [gh*gw, tps*C*ps*ps], gh, gw) → [1, gh*gw/merge², text_hidden].
// ─────────────────────────────────────────────────────────────────────────────

/// A tower linear: quantized (packed weight + scales/biases) or dense bf16.
/// The mirrors ship the tower MIXED — linears quantized, norms and the two
/// gather-read tables bf16 — and an unquantized original must load too, so
/// `.scales` presence is decided PER TENSOR.
const Lin = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array = .{ .ctx = null },
    bits: u32 = 0,
    group: u32 = 0,
    mode: [*:0]const u8 = "affine",
};

const Rope = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

const Block = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    q: Lin,
    k: Lin,
    v: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
};

pub const MuseVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    hidden: u32,
    heads: u32,
    head_dim: u32,
    merge: u32,
    pos_side: u32,
    out_hidden: u32,
    ln_eps: f32,
    rms_eps: f32,
    rope_theta: f64,
    full_attn: [model_mod.MAX_VISION_LAYERS]bool,

    patch_w: mlx.mlx_array,
    pos_table: mlx.mlx_array,
    ln_pre_w: mlx.mlx_array,
    ln_pre_b: mlx.mlx_array,
    ln_post_w: mlx.mlx_array,
    ln_post_b: mlx.mlx_array,
    blocks: []Block,
    fc1: Lin,
    fc2: Lin,
    proj: Lin,
    norm_ones: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !MuseVision {
        const s = mlx.gpuStream();
        // Ours nests the tower under `model.`; mlx-community re-nests it bare.
        const root: []const u8 = if (weights.get("model.vision_tower.ln_pre.weight") != null) "model." else "";
        var name_buf: [160]u8 = undefined;
        const ctx = NameCtx{ .weights = weights, .root = root, .buf = &name_buf, .mode = config.quant_mode.cstr() };

        var blocks = try allocator.alloc(Block, config.qv_depth);
        errdefer allocator.free(blocks);
        for (0..config.qv_depth) |i| {
            blocks[i] = .{
                .norm1_w = try ctx.must("vision_tower.layers.{d}.norm1.weight", .{i}),
                .norm1_b = try ctx.must("vision_tower.layers.{d}.norm1.bias", .{i}),
                .norm2_w = try ctx.must("vision_tower.layers.{d}.norm2.weight", .{i}),
                .norm2_b = try ctx.must("vision_tower.layers.{d}.norm2.bias", .{i}),
                .q = try ctx.lin("vision_tower.layers.{d}.attn.q_proj", .{i}, config.qv_hidden),
                .k = try ctx.lin("vision_tower.layers.{d}.attn.k_proj", .{i}, config.qv_hidden),
                .v = try ctx.lin("vision_tower.layers.{d}.attn.v_proj", .{i}, config.qv_hidden),
                .proj = try ctx.lin("vision_tower.layers.{d}.attn.proj", .{i}, config.qv_hidden),
                .fc1 = try ctx.lin("vision_tower.layers.{d}.mlp.fc1", .{i}, config.qv_hidden),
                .fc2 = try ctx.lin("vision_tower.layers.{d}.mlp.fc2", .{i}, config.qv_intermediate),
            };
        }

        const merged = config.qv_hidden * config.qv_merge * config.qv_merge;
        const one = bf16Scalar(1.0, s);
        defer _ = mlx.mlx_array_free(one);
        const ones_shape = [_]c_int{@intCast(config.hidden_size)};
        var norm_ones = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&norm_ones, &ones_shape, 1, one, .bfloat16, s));

        log.info("Vision encoder: Muse-Glimmer ViT (depth={d}, hidden={d}, heads={d}, merge={d}, out_hidden={d})\n", .{
            config.qv_depth, config.qv_hidden, config.qv_heads, config.qv_merge, config.qv_out_hidden,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .hidden = config.qv_hidden,
            .heads = config.qv_heads,
            .head_dim = config.qv_head_dim,
            .merge = config.qv_merge,
            .pos_side = config.mv_pos_side,
            .out_hidden = config.qv_out_hidden,
            .ln_eps = config.mv_ln_eps,
            .rms_eps = config.rms_norm_eps,
            .rope_theta = config.mv_rope_theta,
            .full_attn = config.mv_full_attn,
            .patch_w = try ctx.must("vision_tower.patch_embedder.patch_embedding.weight", .{}),
            .pos_table = try ctx.must("vision_tower.patch_embedder.position_embedding_table.weight", .{}),
            .ln_pre_w = try ctx.must("vision_tower.ln_pre.weight", .{}),
            .ln_pre_b = try ctx.must("vision_tower.ln_pre.bias", .{}),
            .ln_post_w = try ctx.must("vision_tower.ln_post.weight", .{}),
            .ln_post_b = try ctx.must("vision_tower.ln_post.bias", .{}),
            .blocks = blocks,
            .fc1 = try ctx.lin("vision_adapter.fc1", .{}, merged),
            .fc2 = try ctx.lin("vision_adapter.fc2", .{}, config.mv_projector_hidden),
            .proj = try ctx.lin("vision_projection", .{}, config.mv_projector_hidden),
            .norm_ones = norm_ones,
        };
    }

    pub fn deinit(self: *MuseVision) void {
        _ = mlx.mlx_array_free(self.norm_ones);
        self.allocator.free(self.blocks);
    }

    pub fn forward(self: *MuseVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: c_int = @intCast(grid_h * grid_w);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));
        defer _ = mlx.mlx_array_free(x);

        replace(&x, try self.linear(x, .{ .w = self.patch_w }));
        {
            const pos = try self.posEmbed(grid_h, grid_w);
            defer _ = mlx.mlx_array_free(pos);
            var sum = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&sum, x, pos, self.s));
            replace(&x, sum);
        }
        replace(&x, try self.layerNorm(x, self.ln_pre_w, self.ln_pre_b));

        var win = try windowIndex(self.allocator, grid_h, grid_w, self.pos_side);
        defer win.deinit(self.allocator);
        {
            const idx = hostI32(win.order);
            defer _ = mlx.mlx_array_free(idx);
            var gathered = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(&gathered, x, idx, 0, self.s));
            replace(&x, gathered);
        }

        const rope = try self.buildRope(win.order, grid_w);
        defer _ = mlx.mlx_array_free(rope.cos);
        defer _ = mlx.mlx_array_free(rope.sin);
        const whole = [_]i32{n};

        var dt = mlx.DtypeTrace.begin("muse-vision", x, if (self.blocks.len > 0) self.blocks[0].norm1_w else null);
        for (self.blocks, 0..) |blk, i| {
            {
                const normed = try self.layerNorm(x, blk.norm1_w, blk.norm1_b);
                defer _ = mlx.mlx_array_free(normed);
                const segs: []const i32 = if (self.full_attn[i]) &whole else win.seqlens;
                const attn = try self.attention(normed, blk, rope, segs, n);
                defer _ = mlx.mlx_array_free(attn);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
                replace(&x, h);
            }
            {
                const normed = try self.layerNorm(x, blk.norm2_w, blk.norm2_b);
                defer _ = mlx.mlx_array_free(normed);
                const up = try self.linear(normed, blk.fc1);
                defer _ = mlx.mlx_array_free(up);
                const act = try self.gelu(up);
                defer _ = mlx.mlx_array_free(act);
                const down = try self.linear(act, blk.fc2);
                defer _ = mlx.mlx_array_free(down);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, down, self.s));
                replace(&x, h);
            }
            dt.layer(x, i);
        }
        dt.end(x);

        // Un-window and 2x2-merge in ONE gather: ln_post is row-wise, so it
        // commutes with the permutation the reference applies before it.
        {
            const perm = try self.mergeGather(win.order, grid_h, grid_w);
            defer self.allocator.free(perm);
            const idx = hostI32(perm);
            defer _ = mlx.mlx_array_free(idx);
            var gathered = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(&gathered, x, idx, 0, self.s));
            replace(&x, gathered);
        }
        replace(&x, try self.layerNorm(x, self.ln_post_w, self.ln_post_b));

        // pixel_shuffle: the merged vector is CHANNEL-major over the 2x2 block,
        // i.e. [d0p0, d0p1, d0p2, d0p3, d1p0, …], not patch-major.
        const m2: c_int = @intCast(self.merge * self.merge);
        const n_merged = @divExact(n, m2);
        {
            const grouped = [_]c_int{ n_merged, m2, @intCast(self.hidden) };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, x, &grouped, 3, self.s));
            const axes = [_]c_int{ 0, 2, 1 };
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, r, &axes, 3, self.s));
            _ = mlx.mlx_array_free(r);
            const flat = [_]c_int{ n_merged, @intCast(self.hidden * self.merge * self.merge) };
            var f = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&f, t, &flat, 2, self.s));
            _ = mlx.mlx_array_free(t);
            replace(&x, f);
        }

        // Adapter → projection → weight-less RMSNorm.
        replace(&x, try self.linear(x, self.fc1));
        replace(&x, try self.gelu(x));
        replace(&x, try self.linear(x, self.fc2));
        replace(&x, try self.gelu(x));
        replace(&x, try self.linear(x, self.proj));
        {
            var normed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rms_norm(&normed, x, self.norm_ones, self.rms_eps, self.s));
            replace(&x, normed);
        }

        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, n_merged, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, x, &oshape, 3, self.s));
        return out;
    }

    fn layerNorm(self: *MuseVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, self.ln_eps, self.s));
        return out;
    }

    /// nn.functional.gelu (exact erf form) — both `hidden_act` and
    /// `projector_hidden_act` are plain "gelu".
    fn gelu(self: *MuseVision, x: mlx.mlx_array) !mlx.mlx_array {
        const inv_sqrt2 = bf16Scalar(0.7071067811865476, self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        const one = bf16Scalar(1.0, self.s);
        defer _ = mlx.mlx_array_free(one);
        const half = bf16Scalar(0.5, self.s);
        defer _ = mlx.mlx_array_free(half);
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_multiply(&t, x, inv_sqrt2, self.s));
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_erf(&e, t, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, one, e, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, half, self.s));
        return out;
    }

    fn linear(self: *MuseVision, x: mlx.mlx_array, l: Lin) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        if (l.scales.ctx != null) {
            try mlx.check(mlx.mlx_quantized_matmul(
                &out, x, l.w, l.scales, l.biases, true,
                mlx.mlx_optional_int.some(@intCast(l.group)),
                mlx.mlx_optional_int.some(@intCast(l.bits)),
                l.mode, self.s,
            ));
        } else {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, l.w, self.s));
            try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        }
        if (l.bias.ctx != null) {
            var biased = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&biased, out, l.bias, self.s));
            _ = mlx.mlx_array_free(out);
            out = biased;
        }
        return out;
    }

    /// Learned position table bilinearly resampled onto the image's patch grid,
    /// row-major → [N, hidden].
    fn posEmbed(self: *MuseVision, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const a = self.allocator;
        const n: usize = @as(usize, grid_h) * grid_w;
        var ha = try bilinearAxis(a, grid_h, self.pos_side);
        defer ha.deinit(a);
        var wa = try bilinearAxis(a, grid_w, self.pos_side);
        defer wa.deinit(a);

        var idx: [4][]i32 = undefined;
        var wgt: [4][]f32 = undefined;
        inline for (0..4) |c| {
            idx[c] = try a.alloc(i32, n);
            wgt[c] = try a.alloc(f32, n);
        }
        defer inline for (0..4) |c| {
            a.free(idx[c]);
            a.free(wgt[c]);
        };

        const side: i32 = @intCast(self.pos_side);
        for (0..grid_h) |r| {
            for (0..grid_w) |c| {
                const t = r * grid_w + c;
                const dh = ha.frac[r];
                const dw = wa.frac[c];
                idx[0][t] = ha.lo[r] * side + wa.lo[c];
                idx[1][t] = ha.lo[r] * side + wa.hi[c];
                idx[2][t] = ha.hi[r] * side + wa.lo[c];
                idx[3][t] = ha.hi[r] * side + wa.hi[c];
                wgt[0][t] = if (ha.lo_valid[r] and wa.lo_valid[c]) @floatCast((1 - dh) * (1 - dw)) else 0;
                wgt[1][t] = if (ha.lo_valid[r] and wa.hi_valid[c]) @floatCast((1 - dh) * dw) else 0;
                wgt[2][t] = if (ha.hi_valid[r] and wa.lo_valid[c]) @floatCast(dh * (1 - dw)) else 0;
                wgt[3][t] = if (ha.hi_valid[r] and wa.hi_valid[c]) @floatCast(dh * dw) else 0;
            }
        }

        const wshape = [_]c_int{ @intCast(n), 1 };
        var acc: mlx.mlx_array = undefined;
        inline for (0..4) |c| {
            const ix = hostI32(idx[c]);
            defer _ = mlx.mlx_array_free(ix);
            var gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gathered);
            try mlx.check(mlx.mlx_take_axis(&gathered, self.pos_table, ix, 0, self.s));
            const wf = mlx.mlx_array_new_data(wgt[c].ptr, &wshape, 2, .float32);
            defer _ = mlx.mlx_array_free(wf);
            var wbf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbf);
            try mlx.check(mlx.mlx_astype(&wbf, wf, .bfloat16, self.s));
            var weighted = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&weighted, gathered, wbf, self.s));
            if (c == 0) {
                acc = weighted;
            } else {
                var sum = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&sum, acc, weighted, self.s));
                _ = mlx.mlx_array_free(weighted);
                _ = mlx.mlx_array_free(acc);
                acc = sum;
            }
        }
        return acc;
    }

    /// 2D RoPE tables [N, 1, head_dim] in window order. Both spatial axes use the
    /// FULL head_dim/2 frequency range and interleave as
    /// `[freq_w, freq_h, freq_w, freq_h]`; the reference offsets positions by 1.
    fn buildRope(self: *MuseVision, order: []const i32, grid_w: u32) !Rope {
        const hd: usize = self.head_dim;
        const half = hd / 2;
        const nfreq = half / 2;
        const n = order.len;

        const inv = try self.allocator.alloc(f64, nfreq);
        defer self.allocator.free(inv);
        for (0..nfreq) |k| {
            const e = -@as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(half));
            inv[k] = std.math.pow(f64, self.rope_theta, e);
        }

        const cos_buf = try self.allocator.alloc(f32, n * hd);
        defer self.allocator.free(cos_buf);
        const sin_buf = try self.allocator.alloc(f32, n * hd);
        defer self.allocator.free(sin_buf);

        for (order, 0..) |p, t| {
            const row: f64 = @floatFromInt(@divFloor(p, @as(i32, @intCast(grid_w))) + 1);
            const col: f64 = @floatFromInt(@mod(p, @as(i32, @intCast(grid_w))) + 1);
            const o = t * hd;
            for (0..nfreq) |k| {
                const aw = col * inv[k];
                const ah = row * inv[k];
                inline for (.{ 0, half }) |base| {
                    cos_buf[o + base + k] = @floatCast(@cos(aw));
                    cos_buf[o + base + nfreq + k] = @floatCast(@cos(ah));
                    sin_buf[o + base + k] = @floatCast(@sin(aw));
                    sin_buf[o + base + nfreq + k] = @floatCast(@sin(ah));
                }
            }
        }

        const shape = [_]c_int{ @intCast(n), 1, @intCast(hd) };
        const cf = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(cf);
        const sf = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(sf);
        var cos = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cos, cf, .bfloat16, self.s));
        var sin = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&sin, sf, .bfloat16, self.s));
        return .{ .cos = cos, .sin = sin };
    }

    /// Gather that inverts the window permutation AND applies the 2x2 block
    /// order pixel_shuffle wants, in one pass. Caller owns the result.
    fn mergeGather(self: *MuseVision, order: []const i32, grid_h: u32, grid_w: u32) ![]i32 {
        const n = order.len;
        const inv = try self.allocator.alloc(i32, n);
        defer self.allocator.free(inv);
        for (order, 0..) |p, j| inv[@intCast(p)] = @intCast(j);

        const m = self.merge;
        const out = try self.allocator.alloc(i32, n);
        var t: usize = 0;
        var bh: u32 = 0;
        while (bh < grid_h / m) : (bh += 1) {
            var bw: u32 = 0;
            while (bw < grid_w / m) : (bw += 1) {
                for (0..m) |i| {
                    for (0..m) |j| {
                        const p = (bh * m + i) * grid_w + (bw * m + j);
                        out[t] = inv[p];
                        t += 1;
                    }
                }
            }
        }
        return out;
    }

    fn attention(self: *MuseVision, x: mlx.mlx_array, blk: Block, rope: Rope, segs: []const i32, n: c_int) !mlx.mlx_array {
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = @intCast(self.heads);
        var qkv: [3]mlx.mlx_array = undefined;
        inline for (.{ blk.q, blk.k, blk.v }, 0..) |l, i| {
            const flat = try self.linear(x, l);
            defer _ = mlx.mlx_array_free(flat);
            const shape = [_]c_int{ n, heads, hd };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, flat, &shape, 3, self.s));
            qkv[i] = if (i < 2) blk: {
                defer _ = mlx.mlx_array_free(r);
                break :blk try self.applyRope(r, rope.cos, rope.sin, n, hd);
            } else r;
        }
        defer for (qkv) |a| {
            _ = mlx.mlx_array_free(a);
        };

        // [N, heads, hd] → [1, heads, N, hd] for the fused SDPA.
        var bhnd: [3]mlx.mlx_array = undefined;
        const perm = [_]c_int{ 1, 0, 2 };
        const bshape = [_]c_int{ 1, heads, n, hd };
        inline for (qkv, 0..) |a, i| {
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, a, &perm, 3, self.s));
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&b, t, &bshape, 4, self.s));
            bhnd[i] = b;
        }
        defer for (bhnd) |a| {
            _ = mlx.mlx_array_free(a);
        };

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const ctx = try self.sdpaSegments(bhnd[0], bhnd[1], bhnd[2], segs, scale, heads, hd);
        defer _ = mlx.mlx_array_free(ctx);

        // [1, heads, N, hd] → [N, heads*hd]
        const back = [_]c_int{ 0, 2, 1, 3 };
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx, &back, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const fshape = [_]c_int{ n, @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&flat, t, &fshape, 2, self.s));
        return self.linear(flat, blk.proj);
    }

    /// Block-diagonal attention over `segs` contiguous spans. Window layers see
    /// one span per window; full-attention layers get the whole image as one.
    fn sdpaSegments(self: *MuseVision, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, segs: []const i32, scale: f32, heads: c_int, hd: c_int) !mlx.mlx_array {
        const none = mlx.mlx_array{ .ctx = null };
        if (segs.len == 1) {
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "", none, none, false, self.s));
            return out;
        }
        var parts = try self.allocator.alloc(mlx.mlx_array, segs.len);
        defer self.allocator.free(parts);
        var made: usize = 0;
        errdefer for (parts[0..made]) |p| {
            _ = mlx.mlx_array_free(p);
        };
        var off: c_int = 0;
        for (segs, 0..) |len, i| {
            const stop = off + len;
            const strides = [_]c_int{ 1, 1, 1, 1 };
            const start = [_]c_int{ 0, 0, off, 0 };
            const end = [_]c_int{ 1, heads, stop, hd };
            var sq: [3]mlx.mlx_array = undefined;
            inline for (.{ q, k, v }, 0..) |src, j| {
                var sl = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&sl, src, &start, 4, &end, 4, &strides, 4, self.s));
                sq[j] = sl;
            }
            defer for (sq) |a| {
                _ = mlx.mlx_array_free(a);
            };
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, sq[0], sq[1], sq[2], scale, "", none, none, false, self.s));
            parts[i] = out;
            made += 1;
            off = stop;
        }
        const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 2, self.s));
        for (parts) |p| _ = mlx.mlx_array_free(p);
        return joined;
    }

    /// x·cos + rotate_half(x)·sin over [N, heads, hd]; cos/sin are [N, 1, hd].
    fn applyRope(self: *MuseVision, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int, hd: c_int) !mlx.mlx_array {
        const heads: c_int = @intCast(self.heads);
        const half = @divExact(hd, 2);
        const strides = [_]c_int{ 1, 1, 1 };
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_slice(&x1, x, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ n, heads, half }, 3, &strides, 3, self.s));
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_slice(&x2, x, &[_]c_int{ 0, 0, half }, 3, &[_]c_int{ n, heads, hd }, 3, &strides, 3, self.s));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, x2, self.s));
        const arrs = [_]mlx.mlx_array{ neg, x1 };
        const vec = mlx.mlx_vector_array_new_data(&arrs, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rot);
        try mlx.check(mlx.mlx_concatenate_axis(&rot, vec, -1, self.s));

        var xc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xc);
        try mlx.check(mlx.mlx_multiply(&xc, x, cos, self.s));
        var rs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rs);
        try mlx.check(mlx.mlx_multiply(&rs, rot, sin, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, xc, rs, self.s));
        return out;
    }
};

fn replace(dst: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(dst.*);
    dst.* = next;
}

fn hostI32(v: []const i32) mlx.mlx_array {
    const shape = [_]c_int{@intCast(v.len)};
    return mlx.mlx_array_new_data(v.ptr, &shape, 1, .int32);
}

fn bf16Scalar(v: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&out, f, .bfloat16, s);
    return out;
}

/// Weight lookup under the checkpoint's tower nesting. Handles are BORROWED
/// from the weights map; the encoder owns none of them.
const NameCtx = struct {
    weights: *const Weights,
    root: []const u8,
    buf: *[160]u8,
    mode: [*:0]const u8,

    fn key(self: NameCtx, comptime fmt: []const u8, args: anytype) []const u8 {
        const body = std.fmt.bufPrint(self.buf[80..], fmt, args) catch unreachable;
        return std.fmt.bufPrint(self.buf[0..80], "{s}{s}", .{ self.root, body }) catch unreachable;
    }

    fn opt(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        return self.weights.get(self.key(fmt, args));
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.opt(fmt, args) orelse {
            log.warn("MISSING MUSE VISION WEIGHT: {s}\n", .{self.key(fmt, args)});
            return error.MissingVisionWeights;
        };
    }

    fn lin(self: NameCtx, comptime fmt: []const u8, args: anytype, in_features: u32) !Lin {
        const w = try self.must(fmt ++ ".weight", args);
        var l = Lin{ .w = w, .bias = self.opt(fmt ++ ".bias", args) orelse .{ .ctx = null }, .mode = self.mode };
        if (self.opt(fmt ++ ".scales", args)) |sc| {
            l.scales = sc;
            l.biases = self.opt(fmt ++ ".biases", args) orelse .{ .ctx = null };
            const w_cols: u32 = @intCast(mlx.getShape(w)[1]);
            const s_cols: u32 = @intCast(mlx.getShape(sc)[1]);
            l.bits = @divExact(32 * w_cols, in_features);
            l.group = @divExact(in_features, s_cols);
        }
        return l;
    }
};


// Vision-tower parity vs the EXECUTED reference
// (tests/dump_muse_vision_fixture.py runs transformers' own muse_glimmer vision
// half on OUR dequantized weights, so a diff is a layout/math bug, never
// quantization error).
//
//   MUSE_VISION_MODEL=~/.mlx-serve/models/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit \
//   MUSE_VISION_FIXTURE=~/claude-tmp/muse-vision/muse_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="muse vision parity"
test "muse vision live: tower parity vs the executed reference" {
    const raw_model = std.c.getenv("MUSE_VISION_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("MUSE_VISION_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;

    const config = try model_mod.parseConfig(std.testing.io, a, model_dir);
    var weights = try model_mod.loadWeightsWithVision(std.testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    var mv = try MuseVision.init(a, config, &weights);
    defer mv.deinit();
    const s = mv.s;

    const grid = fx.get("grid_thw") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(grid));
    const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
    const gh: u32 = @intCast(g[1]);
    const gw: u32 = @intCast(g[2]);
    const pv = fx.get("pixel_values") orelse return error.MissingFixtureTensor;

    // Position interpolation first: it runs before every block, so if it
    // differs, nothing downstream can be attributed to the tower.
    {
        const pos = try mv.posEmbed(gh, gw);
        defer _ = mlx.mlx_array_free(pos);
        const want = fx.get("pos_embeds") orelse return error.MissingFixtureTensor;
        const c = try cosineSim(pos, want, s);
        const r = try rmsRatio(pos, want, s);
        std.debug.print("[muse-vit] pos_embeds cos={d:.6} rms_ratio={d:.4}\n", .{ c, r });
        try testing.expect(c > 0.999 and r > 0.99 and r < 1.01);
    }

    const out = try mv.forward(pv, gh, gw);
    defer _ = mlx.mlx_array_free(out);
    const want = fx.get("features") orelse return error.MissingFixtureTensor;
    const c = try cosineSim(out, want, s);
    // MAGNITUDE too: these rows are concatenated into the token stream, where a
    // scale error is exactly the bug a cosine cannot see. (The perception norm
    // pins the RMS at 1, so the ratio also catches a missing norm.)
    const r = try rmsRatio(out, want, s);
    std.debug.print("[muse-vit] features cos={d:.6} rms_ratio={d:.4}\n", .{ c, r });
    try testing.expect(r > 0.99 and r < 1.01);
    // Ours serves bf16 against an fp32 reference through 50 blocks, so the bar
    // is "same features", not bit equality — a layout bug lands far below this.
    try testing.expect(c > 0.99);
}

fn cosineSim(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const dot = try sumSq(a_arr, b_arr, s);
    const na = try sumSq(a_arr, a_arr, s);
    const nb = try sumSq(b_arr, b_arr, s);
    if (!std.math.isFinite(dot) or na <= 0 or nb <= 0) return std.math.nan(f32);
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn rmsRatio(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const na = try sumSq(a_arr, a_arr, s);
    const nb = try sumSq(b_arr, b_arr, s);
    if (!std.math.isFinite(na) or nb <= 0) return std.math.nan(f32);
    return @sqrt(na) / @sqrt(nb);
}

/// sum(a*b) in fp32 over flattened inputs. NaN propagates: `NaN > threshold` is
/// false, so an all-NaN candidate can never pass the comparisons above.
fn sumSq(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a_arr, .float32, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, b_arr, .float32, s));
    const n = [_]c_int{@intCast(mlx.mlx_array_size(af))};
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_reshape(&a1, af, &n, 1, s));
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_reshape(&b1, bf, &n, 1, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, a1, b1, s));
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_sum(&o, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(o));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, o));
    return v;
}

// ── Tests ──

const testing = std.testing;

test "smartResize picks the aspect-closest grid under the merged-token cap" {
    // Values from the reference `smart_resize` (image_processing_muse_glimmer.py)
    // at factor 28 (patch 14 x merge 2), max_tokens 4096.
    const cases = [_]struct { h: u32, w: u32, rh: u32, rw: u32 }{
        .{ .h = 1024, .w = 1024, .rh = 1036, .rw = 1036 },
        .{ .h = 768, .w = 1024, .rh = 756, .rw = 1008 },
        .{ .h = 480, .w = 640, .rh = 476, .rw = 644 },
        .{ .h = 224, .w = 224, .rh = 224, .rw = 224 },
        .{ .h = 100, .w = 37, .rh = 84, .rw = 28 },
        .{ .h = 1, .w = 1, .rh = 28, .rw = 28 },
        // Over the cap: rescaled to fit, then snapped to the closest ratio.
        .{ .h = 4000, .w = 3000, .rh = 2044, .rw = 1540 },
    };
    for (cases) |c| {
        const r = smartResize(c.h, c.w, 28, model_mod.MUSE_MAX_IMAGE_TOKENS);
        try testing.expectEqual(c.rh, r.h);
        try testing.expectEqual(c.rw, r.w);
        // The cap is on MERGED tokens, so it must hold after the grid divides.
        try testing.expect((r.h / 28) * (r.w / 28) <= model_mod.MUSE_MAX_IMAGE_TOKENS);
    }
}

test "windowIndex groups patches window-major with short edge windows" {
    const a = testing.allocator;
    var w = try windowIndex(a, 5, 3, 2);
    defer w.deinit(a);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 3, 4, 2, 5, 6, 7, 9, 10, 8, 11, 12, 13, 14 }, w.order);
    try testing.expectEqualSlices(i32, &.{ 4, 2, 4, 2, 2, 1 }, w.seqlens);

    // Exactly-divisible grid: the reference's `pad = win - gh % win` adds a
    // whole empty window row that unique_consecutive drops — never a 0 seqlen.
    var e = try windowIndex(a, 4, 4, 2);
    defer e.deinit(a);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 }, e.order);
    try testing.expectEqualSlices(i32, &.{ 4, 4, 4, 4 }, e.seqlens);
}

test "buildPixelValues emits row-major patches with [t, c, py, px] features" {
    const a = testing.allocator;
    const patch: u32 = 2;
    const rh: u32 = 4;
    const rw: u32 = 4;
    const C: u32 = 3;
    const tps: u32 = 2;
    // chw[c][y][x] = c*100 + y*10 + x
    var chw: [C * rh * rw]f32 = undefined;
    for (0..C) |c| for (0..rh) |y| for (0..rw) |x| {
        chw[c * rh * rw + y * rw + x] = @floatFromInt(c * 100 + y * 10 + x);
    };
    const feat = tps * C * patch * patch;
    const out = try a.alloc(f32, 4 * feat);
    defer a.free(out);
    buildPixelValues(out, &chw, C, rh, rw, patch, tps);

    // Token 1 is grid (row 0, col 1) — row-major, NOT Qwen's merge-block order.
    const t1 = out[feat .. 2 * feat];
    try testing.expectEqual(@as(f32, 2), t1[0]); // t0 c0 (0,2)
    try testing.expectEqual(@as(f32, 3), t1[1]);
    try testing.expectEqual(@as(f32, 12), t1[2]); // (1,2)
    try testing.expectEqual(@as(f32, 102), t1[4]); // t0 c1
    try testing.expectEqual(@as(f32, 202), t1[8]); // t0 c2
    // The temporal axis is the OUTER one and duplicates the single frame.
    try testing.expectEqualSlices(f32, t1[0 .. feat / 2], t1[feat / 2 ..]);
}

test "bilinearAxis matches grid_sample(align_corners=false) with zero padding" {
    const a = testing.allocator;
    // Upsampling the 32-wide table onto 3 patches: interior only, all valid.
    var up = try bilinearAxis(a, 3, 32);
    defer up.deinit(a);
    try testing.expectEqualSlices(i32, &.{ 4, 15, 26 }, up.lo);
    try testing.expectEqualSlices(i32, &.{ 5, 16, 27 }, up.hi);
    try testing.expectApproxEqAbs(@as(f64, 0.833333), up.frac[0], 1e-5);

    // Downsampling onto 64 patches: the first `lo` and the last `hi` fall
    // OUTSIDE the table — clamped index, zero weight (grid_sample padding).
    var down = try bilinearAxis(a, 64, 32);
    defer down.deinit(a);
    try testing.expect(!down.lo_valid[0]);
    try testing.expect(down.hi_valid[0]);
    try testing.expect(!down.hi_valid[63]);
    try testing.expectEqual(@as(i32, 31), down.hi[63]);
}
