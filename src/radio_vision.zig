//! RADIO tower + nemotron_omni projector for Nemotron 3.5 Bolt.
//!
//! Weights keep the checkpoint names (`vision_model.radio_model.model.*`,
//! `mlp1.*`). The transformers rename in modeling_radio.py is not applied.
//! Images only: a CHW float32 buffer already normalized with CLIP mean/std
//! becomes `[1, tokens, llm_hidden]` after a 32-block ViT, v2 pixel-shuffle,
//! and the ReLU² projector. Class and register tokens are dropped. Video
//! packing is not wired.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const glm_vision = @import("glm_vision.zig");
const log = @import("log.zig");

const ModelConfig = model_mod.ModelConfig;

/// Patch budget. The processor allows 13312, which is a full bidirectional
/// attention over the language model already resident. 4096 patches is 1024
/// language tokens after the 2×2 shuffle.
pub const ENGINE_MAX_PATCHES: u32 = 4096;
pub const MIN_PATCHES: u32 = 1024;
pub const PATCH: u32 = 16;
pub const DOWNSAMPLE: u32 = 2;
const HEADS: u32 = 16;
const LN_EPS: f32 = 1e-6;
const PROJECTOR_EPS: f32 = 1e-5;

/// OPENAI / RADIO CLIP normalization. The tower's input_conditioner is unused;
/// the processor applies this itself.
const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

pub const PatchGrid = struct { w: u32, h: u32 };

/// Python 3 `round` (half to even).
fn roundHalfEven(x: f64) i32 {
    if (!std.math.isFinite(x)) return 0;
    const down = @floor(x);
    const frac = x - down;
    const base: i32 = @intFromFloat(down);
    if (frac < 0.5) return base;
    if (frac > 0.5) return base + 1;
    if (@mod(base, 2) == 0) return base;
    return base + 1;
}

fn snapToDivisor(dim: *u32, other: u32, budget: u32) void {
    const rem = dim.* % DOWNSAMPLE;
    if (rem == 0) return;
    const inc = DOWNSAMPLE - rem;
    if ((dim.* + inc) * other <= budget) {
        dim.* += inc;
    } else {
        dim.* = @max(DOWNSAMPLE, dim.* - rem);
    }
}

/// `(w_patches, h_patches)` from `NemotronH_Omni_Reasoning_V3ImageProcessor._compute_target_patches`.
/// `budget` is a patch count (pre-shuffle), clamped to `[MIN_PATCHES, ENGINE_MAX_PATCHES]`.
pub fn targetPatches(src_h: u32, src_w: u32, budget_in: u32) PatchGrid {
    const budget = std.math.clamp(if (budget_in == 0) ENGINE_MAX_PATCHES else budget_in, MIN_PATCHES, ENGINE_MAX_PATCHES);
    if (src_h == 0 or src_w == 0) return .{ .w = 32, .h = 32 };
    const closest_h: u32 = @intCast(@max(roundHalfEven(@as(f64, @floatFromInt(src_h)) / PATCH + 0.5), 1));
    const closest_w: u32 = @intCast(@max(roundHalfEven(@as(f64, @floatFromInt(src_w)) / PATCH + 0.5), 1));
    const patches: f64 = @floatFromInt(closest_h * closest_w);
    const factor = @min(@sqrt(@as(f64, @floatFromInt(budget)) / patches), 1.0);
    var th: u32 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(closest_h))));
    var tw: u32 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(closest_w))));
    if (th == 0) th = 1;
    if (tw == 0) tw = 1;
    if (budget > MIN_PATCHES and th * tw < MIN_PATCHES) {
        const up = @sqrt(@as(f64, @floatFromInt(MIN_PATCHES)) / @as(f64, @floatFromInt(th * tw)));
        th = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(th))));
        tw = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(tw))));
    }
    snapToDivisor(&th, tw, budget);
    snapToDivisor(&tw, th, budget);
    if (th == 0 or tw == 0) return .{ .w = 32, .h = 32 };
    return .{ .w = tw, .h = th };
}

pub const Prepared = struct {
    pixels: []u8,
    h: u32,
    w: u32,
    tokens: u32,
};

/// PIL bicubic resize (the tower's reference is antialiased torch bicubic) then CLIP norm.
/// Returned bytes are float32 CHW `[3, H, W]`. `tokens` is the language-token count after shuffle.
pub fn prepareImage(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32) !Prepared {
    const grid = targetPatches(src_h, src_w, ENGINE_MAX_PATCHES);
    const tw = grid.w * PATCH;
    const th = grid.h * PATCH;
    const resized = try qwen_vision.resizeInterleavedPil(allocator, rgb, src_h, src_w, th, tw, 3, .bicubic);
    defer allocator.free(resized);
    const plane: usize = @as(usize, th) * tw;
    const buf = try allocator.alloc(u8, plane * 3 * @sizeOf(f32));
    const chw = std.mem.bytesAsSlice(f32, buf);
    for (0..th) |y| {
        for (0..tw) |x| {
            const src = (y * tw + x) * 3;
            const dst = y * tw + x;
            inline for (0..3) |c| {
                const unit = @as(f32, @floatFromInt(resized[src + c])) / 255.0;
                chw[c * plane + dst] = (unit - MEAN[c]) / STD[c];
            }
        }
    }
    return .{
        .pixels = buf,
        .h = th,
        .w = tw,
        .tokens = (grid.h / DOWNSAMPLE) * (grid.w / DOWNSAMPLE),
    };
}

const Block = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fc1_w: mlx.mlx_array,
    fc1_b: mlx.mlx_array,
    fc2_w: mlx.mlx_array,
    fc2_b: mlx.mlx_array,
};

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

fn must(weights: *const Weights, buf: *[320]u8, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return error.MissingVisionWeights;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingVisionWeights;
    };
}

fn scalar(val: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const f = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

pub const RadioVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    prefix: u32,
    pos_side: u32,
    out_hidden: u32,
    pos_owned: bool,
    patch_w: mlx.mlx_array,
    pos_table: mlx.mlx_array,
    cls: mlx.mlx_array,
    blocks: []Block,
    proj_norm: mlx.mlx_array,
    proj_fc1: mlx.mlx_array,
    proj_fc2: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, _: ModelConfig, weights: *const Weights) !RadioVision {
        const s = mlx.gpuStream();
        errdefer _ = mlx.mlx_stream_free(s);
        var buf: [320]u8 = undefined;
        var depth: usize = 0;
        while (depth < 64) : (depth += 1) {
            const name = std.fmt.bufPrint(&buf, "vision_model.radio_model.model.blocks.{d}.attn.qkv.weight", .{depth}) catch break;
            if (weights.get(name) == null) break;
        }
        if (depth == 0) {
            log.err("MISSING WEIGHT: vision_model.radio_model.model.blocks.0.attn.qkv.weight\n", .{});
            return error.MissingVisionWeights;
        }
        const qkv0 = try must(weights, &buf, "vision_model.radio_model.model.blocks.0.attn.qkv.weight", .{});
        const qshape = mlx.getShape(qkv0);
        if (qshape.len != 2 or qshape[0] <= 0 or qshape[1] <= 0) return error.InvalidVisionWeights;
        const hidden: u32 = @intCast(qshape[1]);
        if (@as(u32, @intCast(qshape[0])) != hidden * 3 or hidden % HEADS != 0) return error.InvalidVisionWeights;

        var blocks = try allocator.alloc(Block, depth);
        errdefer allocator.free(blocks);
        for (0..depth) |i| {
            blocks[i] = .{
                .norm1_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm1.weight", .{i}),
                .norm1_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm1.bias", .{i}),
                .norm2_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm2.weight", .{i}),
                .norm2_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm2.bias", .{i}),
                .qkv_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.qkv.weight", .{i}),
                .qkv_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.qkv.bias", .{i}),
                .proj_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.proj.weight", .{i}),
                .proj_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.proj.bias", .{i}),
                .fc1_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc1.weight", .{i}),
                .fc1_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc1.bias", .{i}),
                .fc2_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc2.weight", .{i}),
                .fc2_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc2.bias", .{i}),
            };
        }

        const pos_raw = try must(weights, &buf, "vision_model.radio_model.model.patch_generator.pos_embed", .{});
        const ps = mlx.getShape(pos_raw);
        var pos_table = pos_raw;
        var pos_owned = false;
        if (ps.len == 3 and ps[0] == 1) {
            var flat = mlx.mlx_array_new();
            const shape = [_]c_int{ ps[1], ps[2] };
            try mlx.check(mlx.mlx_reshape(&flat, pos_raw, &shape, 2, s));
            pos_table = flat;
            pos_owned = true;
        }
        errdefer if (pos_owned) {
            _ = mlx.mlx_array_free(pos_table);
        };
        const npos: u32 = @intCast(mlx.getShape(pos_table)[0]);
        const side_f = @sqrt(@as(f64, @floatFromInt(npos)));
        const side: u32 = @intFromFloat(@round(side_f));
        if (side == 0 or side * side != npos) return error.InvalidVisionWeights;

        const cls = try must(weights, &buf, "vision_model.radio_model.model.patch_generator.cls_token.token", .{});
        const cls_shape = mlx.getShape(cls);
        if (cls_shape.len != 2) return error.InvalidVisionWeights;
        const fc2 = try must(weights, &buf, "mlp1.3.weight", .{});
        const fc2_shape = mlx.getShape(fc2);
        if (fc2_shape.len != 2) return error.InvalidVisionWeights;

        log.info("Vision encoder: RADIO (depth={d}, hidden={d}, heads={d}, prefix={d}, pos={d}, out={d}, patch cap={d})\n", .{
            depth, hidden, HEADS, cls_shape[0], side, fc2_shape[0], ENGINE_MAX_PATCHES,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = HEADS,
            .head_dim = hidden / HEADS,
            .prefix = @intCast(cls_shape[0]),
            .pos_side = side,
            .out_hidden = @intCast(fc2_shape[0]),
            .pos_owned = pos_owned,
            .patch_w = try must(weights, &buf, "vision_model.radio_model.model.patch_generator.embedder.weight", .{}),
            .pos_table = pos_table,
            .cls = cls,
            .blocks = blocks,
            .proj_norm = try must(weights, &buf, "mlp1.0.weight", .{}),
            .proj_fc1 = try must(weights, &buf, "mlp1.1.weight", .{}),
            .proj_fc2 = fc2,
        };
    }

    pub fn deinit(self: *RadioVision) void {
        if (self.pos_owned) _ = mlx.mlx_array_free(self.pos_table);
        self.allocator.free(self.blocks);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// pixels `[1, 3, H, W]` float32, H and W divisible by patch and by the shuffle.
    /// Returns `[1, (H/patch/2)*(W/patch/2), out_hidden]`.
    pub fn forward(self: *RadioVision, pixels: mlx.mlx_array) !mlx.mlx_array {
        const shape = mlx.getShape(pixels);
        if (shape.len != 4 or shape[0] != 1 or shape[1] != 3) return error.InvalidVisionInput;
        const height: u32 = @intCast(shape[2]);
        const width: u32 = @intCast(shape[3]);
        if (height % PATCH != 0 or width % PATCH != 0) return error.InvalidVisionInput;
        const rows = height / PATCH;
        const cols = width / PATCH;
        if (rows % DOWNSAMPLE != 0 or cols % DOWNSAMPLE != 0 or rows == 0 or cols == 0) return error.InvalidVisionInput;

        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, pixels, .bfloat16, self.s));
        glm_vision.replace(&x, try self.patchify(x, rows, cols));
        {
            const pos = try self.posEmbed(rows, cols);
            defer _ = mlx.mlx_array_free(pos);
            var summed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&summed, x, pos, self.s));
            glm_vision.replace(&x, summed);
        }
        glm_vision.replace(&x, try self.prependCls(x));

        for (self.blocks) |blk| {
            const normed = try self.layerNorm(x, blk.norm1_w, blk.norm1_b);
            defer _ = mlx.mlx_array_free(normed);
            const attn = try self.attention(normed, blk);
            defer _ = mlx.mlx_array_free(attn);
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
            glm_vision.replace(&x, h);

            const normed2 = try self.layerNorm(x, blk.norm2_w, blk.norm2_b);
            defer _ = mlx.mlx_array_free(normed2);
            const up = try self.dense(normed2, blk.fc1_w, blk.fc1_b);
            defer _ = mlx.mlx_array_free(up);
            const act = try glm_vision.geluErf(up, self.s);
            defer _ = mlx.mlx_array_free(act);
            const down = try self.dense(act, blk.fc2_w, blk.fc2_b);
            defer _ = mlx.mlx_array_free(down);
            var h2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h2, x, down, self.s));
            glm_vision.replace(&x, h2);
        }

        glm_vision.replace(&x, try self.dropPrefix(x));
        glm_vision.replace(&x, try self.pixelShuffle(x, rows, cols));
        glm_vision.replace(&x, try self.projector(x));
        return x;
    }

    fn patchify(self: *RadioVision, x: mlx.mlx_array, rows: u32, cols: u32) !mlx.mlx_array {
        const ps: c_int = PATCH;
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        const in_shape = [_]c_int{ 1, 3, @intCast(rows), ps, @intCast(cols), ps };
        try mlx.check(mlx.mlx_reshape(&r, x, &in_shape, 6, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        const axes = [_]c_int{ 0, 2, 4, 1, 3, 5 };
        try mlx.check(mlx.mlx_transpose_axes(&t, r, &axes, 6, self.s));
        const flat = [_]c_int{ 1, @intCast(rows * cols), 3 * ps * ps };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, t, &flat, 3, self.s));
        const projected = try self.dense(out, self.patch_w, .{ .ctx = null });
        _ = mlx.mlx_array_free(out);
        return projected;
    }

    fn prependCls(self: *RadioVision, x: mlx.mlx_array) !mlx.mlx_array {
        var cls = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cls);
        const shape = [_]c_int{ 1, @intCast(self.prefix), @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&cls, self.cls, &shape, 3, self.s));
        const parts = [_]mlx.mlx_array{ cls, x };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, self.s));
        return out;
    }

    fn dropPrefix(self: *RadioVision, x: mlx.mlx_array) !mlx.mlx_array {
        const shape = mlx.getShape(x);
        const start = [_]c_int{ 0, @intCast(self.prefix), 0 };
        const stop = [_]c_int{ shape[0], shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&out, x, &start, 3, &stop, 3, &strides, 3, self.s));
        return out;
    }

    /// Cropped position embedding: bilinear to max(rows, cols), then the top-left crop.
    fn posEmbed(self: *RadioVision, rows: u32, cols: u32) !mlx.mlx_array {
        const a = self.allocator;
        const side = self.pos_side;
        const max_dim = @max(rows, cols);
        var ha = try bilinearAxis(a, max_dim, side);
        defer ha.deinit(a);
        var wa = try bilinearAxis(a, max_dim, side);
        defer wa.deinit(a);

        const n: usize = @as(usize, rows) * cols;
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

        const side_i: i32 = @intCast(side);
        for (0..rows) |r| {
            for (0..cols) |c| {
                const t = r * cols + c;
                const dh = ha.frac[r];
                const dw = wa.frac[c];
                idx[0][t] = ha.lo[r] * side_i + wa.lo[c];
                idx[1][t] = ha.lo[r] * side_i + wa.hi[c];
                idx[2][t] = ha.hi[r] * side_i + wa.lo[c];
                idx[3][t] = ha.hi[r] * side_i + wa.hi[c];
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
        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, @intCast(n), @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&out, acc, &oshape, 3, self.s));
        _ = mlx.mlx_array_free(acc);
        return out;
    }

    fn attention(self: *RadioVision, x: mlx.mlx_array, blk: Block) !mlx.mlx_array {
        const shape = mlx.getShape(x);
        const seq = shape[1];
        const heads: c_int = @intCast(self.heads);
        const hd: c_int = @intCast(self.head_dim);
        const qkv = try self.dense(x, blk.qkv_w, blk.qkv_b);
        defer _ = mlx.mlx_array_free(qkv);
        var parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        try mlx.check(mlx.mlx_split(&parts, qkv, 3, 2, self.s));

        var qkv_h: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer {
            for (qkv_h) |a| _ = mlx.mlx_array_free(a);
        }
        for (0..3) |j| {
            var piece = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(piece);
            try mlx.check(mlx.mlx_vector_array_get(&piece, parts, j));
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_reshape(&r, piece, &[_]c_int{ 1, seq, heads, hd }, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&qkv_h[j], r, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array{ .ctx = null };
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx, qkv_h[0], qkv_h[1], qkv_h[2], scale, "", none, none, false, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ 1, seq, @intCast(self.hidden) }, 3, self.s));
        return self.dense(flat, blk.proj_w, blk.proj_b);
    }

    fn pixelShuffle(self: *RadioVision, x: mlx.mlx_array, rows: u32, cols: u32) !mlx.mlx_array {
        const c: c_int = @intCast(self.hidden);
        var grid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grid);
        try mlx.check(mlx.mlx_reshape(&grid, x, &[_]c_int{ 1, @intCast(rows), @intCast(cols), c }, 4, self.s));
        var v1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v1);
        try mlx.check(mlx.mlx_reshape(&v1, grid, &[_]c_int{ 1, @intCast(rows), @intCast(cols / 2), c * 2 }, 4, self.s));
        var p1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p1);
        try mlx.check(mlx.mlx_transpose_axes(&p1, v1, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        var v2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v2);
        try mlx.check(mlx.mlx_reshape(&v2, p1, &[_]c_int{ 1, @intCast(cols / 2), @intCast(rows / 2), c * 4 }, 4, self.s));
        var p2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p2);
        try mlx.check(mlx.mlx_transpose_axes(&p2, v2, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        const ntok: c_int = @intCast((rows / 2) * (cols / 2));
        var flat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&flat, p2, &[_]c_int{ 1, ntok, c * 4 }, 3, self.s));
        return flat;
    }

    fn projector(self: *RadioVision, x: mlx.mlx_array) !mlx.mlx_array {
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, x, self.proj_norm, PROJECTOR_EPS, self.s));
        const up = try self.dense(normed, self.proj_fc1, .{ .ctx = null });
        defer _ = mlx.mlx_array_free(up);
        const act = try reluSquared(up, self.s);
        defer _ = mlx.mlx_array_free(act);
        return self.dense(act, self.proj_fc2, .{ .ctx = null });
    }

    fn layerNorm(self: *RadioVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, LN_EPS, self.s));
        return out;
    }

    fn dense(self: *RadioVision, x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
        return glm_vision.linear(x, .{ .w = w, .bias = bias }, self.s);
    }
};

fn reluSquared(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const zero = try scalar(0.0, mlx.mlx_array_dtype(x), s);
    defer _ = mlx.mlx_array_free(zero);
    var relu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(relu);
    try mlx.check(mlx.mlx_maximum(&relu, x, zero, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_square(&out, relu, s));
    return out;
}

fn hostI32(v: []const i32) mlx.mlx_array {
    const shape = [_]c_int{@intCast(v.len)};
    return mlx.mlx_array_new_data(v.ptr, &shape, 1, .int32);
}

test "radio targetPatches matches the processor on a 512 square" {
    // 512/16 is an integer. Python round(32.5) is 32, so the grid stays 32×32
    // and the shuffle yields 256 language tokens. The engine cap is above the minimum.
    const g = targetPatches(512, 512, ENGINE_MAX_PATCHES);
    try std.testing.expectEqual(@as(u32, 32), g.w);
    try std.testing.expectEqual(@as(u32, 32), g.h);
}

test "radio targetPatches keeps a wide image inside the patch cap" {
    const g = targetPatches(3000, 4000, ENGINE_MAX_PATCHES);
    try std.testing.expect(g.w % 2 == 0 and g.h % 2 == 0);
    try std.testing.expect(g.w * g.h <= ENGINE_MAX_PATCHES);
    try std.testing.expect(g.w * g.h >= MIN_PATCHES);
}
