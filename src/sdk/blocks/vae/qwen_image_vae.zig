//! Qwen-Image's 3D causal VAE (diffusers `AutoencoderKLQwenImage`, the Wan 2.1 layout: 16 latent channels, /8) run
//! one frame at a time, so every causal conv3d is the conv2d of its last temporal tap and an `upsample3d` is an
//! `upsample2d` (the first frame skips its time conv). Channels-last, computed in `dt`. Krea and Ming-Image; the
//! output channels come off `conv_out` (3, or 4 for Ming's RGBA). Raw diffusers tensor names, transposed at load.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../util/log.zig");
const weights_load = @import("../../weights/load.zig");
const conv_blk = @import("conv.zig");
const resnet_blk = @import("resnet.zig");

const Weights = weights_load.Weights;
const S = mlx.mlx_stream;
const Graph = ops.Graph;
const Array = ops.Array;
const Resnet = resnet_blk.Resnet;

/// RMSNorm over channels (x / max(‖x‖₂, eps) · √C · γ = rms_norm over the last axis of NHWC).
const norm: resnet_blk.Norm = .{ .rms = 1e-12 };

/// How a latent maps to the VAE's own space: per-channel statistics (z·std + mean) or one scale and shift
/// (z / factor + shift).
pub const Latents = union(enum) {
    per_channel: struct { mean: []const f32, std: []const f32 },
    scaled: struct { factor: f32, shift: f32 = 0 },
};

fn stat(g: *Graph, vals: []const f32, dt: ops.Dtype) !Array {
    var c = g.scope();
    defer c.deinit();
    return g.keep(try c.astype(try c.fromSlice(f32, vals, &.{@intCast(vals.len)}), dt));
}

fn freeResnet(r: *Resnet) void {
    inline for (.{ "n1w", "c1w", "c1b", "n2w", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(r, f));
    inline for (.{ "sw", "sb" }) |f| if (@field(r, f)) |x| {
        _ = mlx.mlx_array_free(x);
    };
}

const MidAttn = struct {
    norm: mlx.mlx_array,
    qkv_w: mlx.mlx_array, // conv2d 1x1 [3C,1,1,C]
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fn deinit(self: *MidAttn) void {
        inline for (.{ "norm", "qkv_w", "qkv_b", "proj_w", "proj_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
    fn forward(self: *const MidAttn, g: *Graph, x: Array) !Array {
        return resnet_blk.midAttentionQkv(g, norm, x, self.norm, null, self.qkv_w, self.qkv_b, self.proj_w, self.proj_b);
    }
};

const UpBlock = struct {
    resnets: []Resnet,
    up_w: ?mlx.mlx_array = null, // resample conv2d [out,kh,kw,in]
    up_b: ?mlx.mlx_array = null,
    allocator: std.mem.Allocator,
    fn deinit(self: *UpBlock) void {
        for (self.resnets) |*r| freeResnet(r);
        self.allocator.free(self.resnets);
        if (self.up_w) |x| _ = mlx.mlx_array_free(x);
        if (self.up_b) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const UpBlock, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        var h = x;
        for (self.resnets) |*r| h = try r.forward(&g, norm, h);
        if (self.up_w) |uw| h = try conv_blk.upsample2x(&g, h, uw, self.up_b.?);
        return parent.keep(h);
    }
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    dt: ops.Dtype,
    latents: Latents,
    pq_w: mlx.mlx_array, // post_quant_conv (1x1x1)
    pq_b: mlx.mlx_array,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: MidAttn,
    mid_r1: Resnet,
    up_blocks: [4]UpBlock,
    norm_out: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Decoder) void {
        inline for (.{ "pq_w", "pq_b", "conv_in_w", "conv_in_b", "norm_out", "conv_out_w", "conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        freeResnet(&self.mid_r0);
        self.mid_attn.deinit();
        freeResnet(&self.mid_r1);
        for (&self.up_blocks) |*b| b.deinit();
    }

    /// latent [1,16,lat_h,lat_w] f32 → image [1,C,8·lat_h,8·lat_w] f32 in about [-1,1] (unclamped).
    pub fn decode(self: *const Decoder, parent: *Graph, latent: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const l = try g.astype(try g.transpose(latent, &.{ 0, 2, 3, 1 }), self.dt); // NHWC
        const z = switch (self.latents) {
            .per_channel => |p| try g.add(try g.mul(l, try stat(&g, p.std, self.dt)), try stat(&g, p.mean, self.dt)),
            .scaled => |sc| blk: {
                const d = try g.div(l, try g.scalar(sc.factor, self.dt));
                break :blk if (sc.shift == 0) d else try g.add(d, try g.scalar(sc.shift, self.dt));
            },
        };
        var h = try conv_blk.conv2d(&g, z, self.pq_w, self.pq_b, 1, 0, 1);
        h = try conv_blk.conv2d(&g, h, self.conv_in_w, self.conv_in_b, 1, 1, 1);
        h = try self.mid_r0.forward(&g, norm, h);
        h = try self.mid_attn.forward(&g, h);
        h = try self.mid_r1.forward(&g, norm, h);
        for (&self.up_blocks) |*b| h = try b.forward(&g, h);
        h = try g.silu(try g.rmsNorm(h, self.norm_out, 1e-12));
        h = try conv_blk.conv2d(&g, h, self.conv_out_w, self.conv_out_b, 1, 1, 1);
        return parent.keep(try g.contiguous(try g.astype(try g.transpose(h, &.{ 0, 3, 1, 2 }), .float32)));
    }
};

/// One entry of the encoder's flat down_blocks list (resnets at 0,1,3,4,6,7,9,10; spatial resamples at 2,5,8).
const EncBlock = union(enum) {
    res: Resnet,
    down: struct { w: mlx.mlx_array, b: mlx.mlx_array },
};

/// The encoder (T=1), the decoder's mirror: pixels → normalized latents, the space `Decoder.decode` reads.
pub const Encoder = struct {
    dt: ops.Dtype,
    latents: Latents,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    blocks: [11]EncBlock,
    mid_r0: Resnet,
    mid_attn: MidAttn,
    mid_r1: Resnet,
    norm_out: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    quant_w: mlx.mlx_array,
    quant_b: mlx.mlx_array,

    pub fn deinit(self: *Encoder) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out", "conv_out_w", "conv_out_b", "quant_w", "quant_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        for (&self.blocks) |*blk| switch (blk.*) {
            .res => |*r| freeResnet(r),
            .down => |*d| {
                _ = mlx.mlx_array_free(d.w);
                _ = mlx.mlx_array_free(d.b);
            },
        };
        freeResnet(&self.mid_r0);
        self.mid_attn.deinit();
        freeResnet(&self.mid_r1);
    }

    /// image [1,C,H,W] f32 [0,1] (NCHW, H/W multiples of 16) → latents [1,16,H/8,W/8] f32, normalized (the
    /// distribution mean: deterministic, no sampling).
    pub fn encode(self: *const Encoder, parent: *Graph, img: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const signed = try g.sub(try g.mul(img, try g.scalar(2.0, .float32)), try g.scalar(1.0, .float32));
        var h = try g.astype(try g.transpose(signed, &.{ 0, 2, 3, 1 }), self.dt);
        h = try conv_blk.conv2d(&g, h, self.conv_in_w, self.conv_in_b, 1, 1, 1);
        for (&self.blocks) |*blk| h = switch (blk.*) {
            .res => |*rb| try rb.forward(&g, norm, h),
            .down => |*d| try conv_blk.downsample2x(&g, h, d.w, d.b),
        };
        h = try self.mid_r0.forward(&g, norm, h);
        h = try self.mid_attn.forward(&g, h);
        h = try self.mid_r1.forward(&g, norm, h);
        h = try g.silu(try g.rmsNorm(h, self.norm_out, 1e-12));
        const co = try conv_blk.conv2d(&g, h, self.conv_out_w, self.conv_out_b, 1, 1, 1); // [1,h,w,2z]
        const qc = try conv_blk.conv2d(&g, co, self.quant_w, self.quant_b, 1, 0, 1);
        const z_dim = @divExact(ops.shape(qc)[3], 2);
        const mf = try g.astype(try g.sliceAxis(qc, 3, 0, z_dim), .float32); // the mean; logvar discarded
        const zn = switch (self.latents) {
            .per_channel => |p| try g.div(try g.sub(mf, try stat(&g, p.mean, .float32)), try stat(&g, p.std, .float32)),
            .scaled => |sc| try g.mul(try g.sub(mf, try g.scalar(sc.shift, .float32)), try g.scalar(sc.factor, .float32)),
        };
        return parent.keep(try g.contiguous(try g.transpose(zn, &.{ 0, 3, 1, 2 })));
    }
};

// ── loading ──

const Loader = struct {
    w: *const Weights,
    a: std.mem.Allocator,
    s: S,
    dt: ops.Dtype,
    buf: [192]u8 = undefined,

    fn key(l: *Loader, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&l.buf, f, args);
    }

    fn raw(l: *Loader, comptime f: []const u8, args: anytype) !mlx.mlx_array {
        const k = try l.key(f, args);
        return l.w.get(k) orelse {
            log.err("[vae] MISSING WEIGHT: {s}\n", .{k});
            return error.MissingVaeWeight;
        };
    }

    fn has(l: *Loader, comptime f: []const u8, args: anytype) bool {
        return l.w.get(l.key(f, args) catch return false) != null;
    }

    fn finish(l: *Loader, x: mlx.mlx_array) !mlx.mlx_array {
        defer _ = mlx.mlx_array_free(x);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, x, false, l.s));
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, c, l.dt, l.s));
        return o;
    }

    /// A causal conv3d weight as the conv2d it is over one frame: [out,in,kt,kh,kw] → its last temporal tap as
    /// [out,kh,kw,in].
    fn conv3d(l: *Loader, comptime f: []const u8, args: anytype) !mlx.mlx_array {
        const r = try l.raw(f ++ ".weight", args);
        const sh = mlx.getShape(r);
        const kt = sh[2];
        var tap = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tap);
        const lo = [_]c_int{ 0, 0, kt - 1, 0, 0 };
        const hi = [_]c_int{ sh[0], sh[1], kt, sh[3], sh[4] };
        const st = [_]c_int{ 1, 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&tap, r, &lo, 5, &hi, 5, &st, 5, l.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, tap, &[_]c_int{ 0, 2, 3, 4, 1 }, 5, l.s)); // [out,1,kh,kw,in]
        var t4 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&t4, t, &[_]c_int{ sh[0], sh[3], sh[4], sh[1] }, 4, l.s));
        return l.finish(t4);
    }

    /// A conv2d weight [out,in,kh,kw] → [out,kh,kw,in].
    fn conv2d(l: *Loader, comptime f: []const u8, args: anytype) !mlx.mlx_array {
        var t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&t, try l.raw(f ++ ".weight", args), &[_]c_int{ 0, 2, 3, 1 }, 4, l.s));
        return l.finish(t);
    }

    fn vec(l: *Loader, comptime f: []const u8, args: anytype) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, try l.raw(f, args), l.dt, l.s));
        return o;
    }

    /// A `.gamma` RMSNorm vector, flattened to [C].
    fn gamma(l: *Loader, comptime f: []const u8, args: anytype) !mlx.mlx_array {
        const r = try l.raw(f ++ ".gamma", args);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, r, &[_]c_int{mlx.getShape(r)[0]}, 1, l.s));
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, flat, l.dt, l.s));
        return o;
    }

    fn resnet(l: *Loader, comptime f: []const u8, args: anytype) !Resnet {
        var r: Resnet = .{
            .n1w = try l.gamma(f ++ ".norm1", args),
            .c1w = try l.conv3d(f ++ ".conv1", args),
            .c1b = try l.vec(f ++ ".conv1.bias", args),
            .n2w = try l.gamma(f ++ ".norm2", args),
            .c2w = try l.conv3d(f ++ ".conv2", args),
            .c2b = try l.vec(f ++ ".conv2.bias", args),
        };
        if (l.has(f ++ ".conv_shortcut.weight", args)) {
            r.sw = try l.conv3d(f ++ ".conv_shortcut", args);
            r.sb = try l.vec(f ++ ".conv_shortcut.bias", args);
        }
        return r;
    }

    fn mid(l: *Loader, comptime side: []const u8) !MidAttn {
        const p = side ++ ".mid_block.attentions.0";
        return .{
            .norm = try l.gamma(p ++ ".norm", .{}),
            .qkv_w = try l.conv2d(p ++ ".to_qkv", .{}),
            .qkv_b = try l.vec(p ++ ".to_qkv.bias", .{}),
            .proj_w = try l.conv2d(p ++ ".proj", .{}),
            .proj_b = try l.vec(p ++ ".proj.bias", .{}),
        };
    }
};

fn loadWeights(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Weights {
    const dir = try std.fmt.allocPrint(a, "{s}/vae", .{model_dir});
    defer a.free(dir);
    return weights_load.loadWeights(io, a, dir);
}

/// `<model_dir>/vae`'s decoder, computed in `dt`.
pub fn loadDecoder(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, dt: ops.Dtype, latents: Latents) !Decoder {
    var w = try loadWeights(io, a, model_dir);
    defer w.deinit();
    var l: Loader = .{ .w = &w, .a = a, .s = s, .dt = dt };
    var v: Decoder = undefined;
    v.allocator = a;
    v.dt = dt;
    v.latents = latents;
    v.pq_w = try l.conv3d("post_quant_conv", .{});
    v.pq_b = try l.vec("post_quant_conv.bias", .{});
    v.conv_in_w = try l.conv3d("decoder.conv_in", .{});
    v.conv_in_b = try l.vec("decoder.conv_in.bias", .{});
    v.mid_r0 = try l.resnet("decoder.mid_block.resnets.0", .{});
    v.mid_r1 = try l.resnet("decoder.mid_block.resnets.1", .{});
    v.mid_attn = try l.mid("decoder");
    // up_blocks 0..2 upsample (resample.1 conv); 3 has no upsampler.
    for (0..4) |bi| {
        var ub: UpBlock = .{ .resnets = try a.alloc(Resnet, 3), .allocator = a };
        for (0..3) |ri| ub.resnets[ri] = try l.resnet("decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
        if (bi < 3) {
            ub.up_w = try l.conv2d("decoder.up_blocks.{d}.upsamplers.0.resample.1", .{bi});
            ub.up_b = try l.vec("decoder.up_blocks.{d}.upsamplers.0.resample.1.bias", .{bi});
        }
        v.up_blocks[bi] = ub;
    }
    v.norm_out = try l.gamma("decoder.norm_out", .{});
    v.conv_out_w = try l.conv3d("decoder.conv_out", .{});
    v.conv_out_b = try l.vec("decoder.conv_out.bias", .{});
    return v;
}

/// `<model_dir>/vae`'s encoder, computed in `dt`.
pub fn loadEncoder(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, dt: ops.Dtype, latents: Latents) !Encoder {
    var w = try loadWeights(io, a, model_dir);
    defer w.deinit();
    var l: Loader = .{ .w = &w, .a = a, .s = s, .dt = dt };
    var v: Encoder = undefined;
    v.dt = dt;
    v.latents = latents;
    v.conv_in_w = try l.conv3d("encoder.conv_in", .{});
    v.conv_in_b = try l.vec("encoder.conv_in.bias", .{});
    for (0..11) |bi| {
        v.blocks[bi] = if (bi == 2 or bi == 5 or bi == 8) .{ .down = .{
            .w = try l.conv2d("encoder.down_blocks.{d}.resample.1", .{bi}),
            .b = try l.vec("encoder.down_blocks.{d}.resample.1.bias", .{bi}),
        } } else .{ .res = try l.resnet("encoder.down_blocks.{d}", .{bi}) };
    }
    v.mid_r0 = try l.resnet("encoder.mid_block.resnets.0", .{});
    v.mid_r1 = try l.resnet("encoder.mid_block.resnets.1", .{});
    v.mid_attn = try l.mid("encoder");
    v.norm_out = try l.gamma("encoder.norm_out", .{});
    v.conv_out_w = try l.conv3d("encoder.conv_out", .{});
    v.conv_out_b = try l.vec("encoder.conv_out.bias", .{});
    v.quant_w = try l.conv3d("quant_conv", .{});
    v.quant_b = try l.vec("quant_conv.bias", .{});
    return v;
}
