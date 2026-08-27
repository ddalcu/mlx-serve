//! FLUX.1 VAE decoder — diffusers AutoencoderKL, 16 latent channels (vs FLUX.2's
//! batch-norm/128-channel VAE in flux.zig, which does not generalize). Ported to
//! mlx-c FFI. Convs are bf16 in MLX OHWI layout; attention q/k/v/out are affine
//! quantized. See docs/reference.md (FLUX.1 section).
//!
//! mflux wraps the top-level convs (`conv_in.conv2d`, `conv_out.conv2d`) and the
//! output norm (`conv_norm_out.norm`) but not the resnet internals — handled at
//! load. There is NO post_quant_conv in the pack. Latents are denormalized as
//! `z = latents / scaling_factor + shift_factor` before conv_in.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const fc = @import("flux_common.zig");

const QLinear = fc.QLinear;
const S = fc.S;
const Weights = model_mod.Weights;
const addA = fc.addA;
const mulA = fc.mulA;
const subA = fc.subA;
const reshape = fc.reshape;
const transpose = fc.transpose;
const astype = fc.astype;
const silu = fc.silu;
const ownWeight = fc.ownWeight;
const ownOpt = fc.ownOpt;
const fmtKey = fc.fmtKey;

// FLUX.1 AutoencoderKL constants.
const SCALING_FACTOR: f32 = 0.3611;
const SHIFT_FACTOR: f32 = 0.1159;
const NORM_GROUPS: c_int = 32;

// ── conv / norm helpers (VAE-local; mirror flux.zig's private ones) ──

fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,W,C]
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const r1 = try reshape(xf, &[_]c_int{ 1, H * Wd, groups, cg }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, H * Wd * cg }, s);
    defer _ = mlx.mlx_array_free(flat);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer _ = mlx.mlx_array_free(norm);
    const b1 = try reshape(norm, &[_]c_int{ 1, groups, H * Wd, cg }, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(b3);
    const wf = try astype(weight, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(bias, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const sc = try mulA(b3, wf, s);
    defer _ = mlx.mlx_array_free(sc);
    const out = try addA(sc, bf, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}

const Resnet = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    sw: ?mlx.mlx_array = null,
    sb: ?mlx.mlx_array = null,
    fn deinit(self: *Resnet) void {
        inline for (.{ "n1w", "n1b", "c1w", "c1b", "n2w", "n2b", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.sw) |x| _ = mlx.mlx_array_free(x);
        if (self.sb) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const Resnet, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try groupNorm(x, self.n1w, self.n1b, NORM_GROUPS, s);
        defer _ = mlx.mlx_array_free(h0);
        const a0 = try silu(h0, s);
        defer _ = mlx.mlx_array_free(a0);
        const c1 = try conv2d(a0, self.c1w, self.c1b, 1, s);
        defer _ = mlx.mlx_array_free(c1);
        const h1 = try groupNorm(c1, self.n2w, self.n2b, NORM_GROUPS, s);
        defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s);
        defer _ = mlx.mlx_array_free(a1);
        const c2 = try conv2d(a1, self.c2w, self.c2b, 1, s);
        defer _ = mlx.mlx_array_free(c2);
        if (self.sw) |sw| {
            const sc = try conv2d(x, sw, self.sb.?, 0, s);
            defer _ = mlx.mlx_array_free(sc);
            return addA(c2, sc, s);
        }
        return addA(c2, x, s);
    }
};

const VaeAttn = struct {
    gnw: mlx.mlx_array,
    gnb: mlx.mlx_array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    fn deinit(self: *VaeAttn) void {
        _ = mlx.mlx_array_free(self.gnw);
        _ = mlx.mlx_array_free(self.gnb);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
    }
    fn forward(self: *const VaeAttn, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const H = sh[1];
        const Wd = sh[2];
        const C = sh[3];
        const normed = try groupNorm(x, self.gnw, self.gnb, NORM_GROUPS, s);
        defer _ = mlx.mlx_array_free(normed);
        const q = try self.q.forward(normed, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.k.forward(normed, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.v.forward(normed, s);
        defer _ = mlx.mlx_array_free(v);
        const qr = try reshape(q, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try reshape(k, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try reshape(v, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(vr);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vr, scale, "", null_a, null_a, false, s));
        const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s);
        defer _ = mlx.mlx_array_free(ar);
        const ao = try self.o.forward(ar, s);
        defer _ = mlx.mlx_array_free(ao);
        return addA(x, ao, s);
    }
};

pub const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: VaeAttn,
    mid_r1: Resnet,
    up_resnets: [4][3]Resnet,
    up_conv_w: [3]mlx.mlx_array,
    up_conv_b: [3]mlx.mlx_array,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Vae) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
        for (&self.up_resnets) |*blk| for (blk) |*r| r.deinit();
        for (0..3) |i| {
            _ = mlx.mlx_array_free(self.up_conv_w[i]);
            _ = mlx.mlx_array_free(self.up_conv_b[i]);
        }
    }

    /// latents [1,16,h,w] (NCHW) → image [1,3,H,W] (NCHW, [-1,1]).
    pub fn decode(self: *Vae, latents: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        // Denormalize: z = latents / scaling_factor + shift_factor.
        const inv_sf = mlx.mlx_array_new_float(1.0 / SCALING_FACTOR);
        defer _ = mlx.mlx_array_free(inv_sf);
        const shf = mlx.mlx_array_new_float(SHIFT_FACTOR);
        defer _ = mlx.mlx_array_free(shf);
        const lf = try astype(latents, .float32, s);
        defer _ = mlx.mlx_array_free(lf);
        const scaled = try mulA(lf, inv_sf, s);
        defer _ = mlx.mlx_array_free(scaled);
        const z = try addA(scaled, shf, s);
        defer _ = mlx.mlx_array_free(z);
        const z_bf = try astype(z, .bfloat16, s);
        defer _ = mlx.mlx_array_free(z_bf);
        // NCHW → NHWC for conv.
        const nhwc = try transpose(z_bf, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc);
        var h = try conv2d(nhwc, self.conv_in_w, self.conv_in_b, 1, s);
        // mid block
        {
            const nh = try self.mid_r0.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_attn.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_r1.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        // up blocks (3 resnets each; upsample on blocks 0,1,2)
        for (0..4) |bi| {
            for (0..3) |ri| {
                const nh = try self.up_resnets[bi][ri].forward(h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (bi < 3) {
                const us = try self.upsample(h, self.up_conv_w[bi], self.up_conv_b[bi], s);
                _ = mlx.mlx_array_free(h);
                h = us;
            }
        }
        {
            const nh = try groupNorm(h, self.norm_out_w, self.norm_out_b, NORM_GROUPS, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try silu(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.conv_out_w, self.conv_out_b, 1, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        const out = try transpose(h, &[_]c_int{ 0, 3, 1, 2 }, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(out);
        var contig = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&contig, out, false, s));
        return contig;
    }

    fn upsample(self: *Vae, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
        _ = self;
        var r1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r1);
        try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
        var r2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r2);
        try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
        return conv2d(r2, w, b, 1, s);
    }
};

fn loadResnet(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !Resnet {
    const g = struct {
        fn k(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
            const kk = try fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(kk);
            return ownWeight(ww, kk);
        }
        fn opt(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) ?mlx.mlx_array {
            const kk = fmtKey(aa, "{s}.{s}", .{ p, sub }) catch return null;
            defer aa.free(kk);
            return ownOpt(ww, kk);
        }
    };
    return .{
        .n1w = try g.k(w, a, pfx, "norm1.weight"),
        .n1b = try g.k(w, a, pfx, "norm1.bias"),
        .c1w = try g.k(w, a, pfx, "conv1.weight"),
        .c1b = try g.k(w, a, pfx, "conv1.bias"),
        .n2w = try g.k(w, a, pfx, "norm2.weight"),
        .n2b = try g.k(w, a, pfx, "norm2.bias"),
        .c2w = try g.k(w, a, pfx, "conv2.weight"),
        .c2b = try g.k(w, a, pfx, "conv2.bias"),
        .sw = g.opt(w, a, pfx, "conv_shortcut.weight"),
        .sb = g.opt(w, a, pfx, "conv_shortcut.bias"),
    };
}

fn loadAttn(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !VaeAttn {
    const gnw = try fmtKey(a, "{s}.group_norm.weight", .{pfx});
    defer a.free(gnw);
    const gnb = try fmtKey(a, "{s}.group_norm.bias", .{pfx});
    defer a.free(gnb);
    const qp = try fmtKey(a, "{s}.to_q", .{pfx});
    defer a.free(qp);
    const kp = try fmtKey(a, "{s}.to_k", .{pfx});
    defer a.free(kp);
    const vp = try fmtKey(a, "{s}.to_v", .{pfx});
    defer a.free(vp);
    const op = try fmtKey(a, "{s}.to_out.0", .{pfx});
    defer a.free(op);
    return .{
        .gnw = try ownWeight(w, gnw),
        .gnb = try ownWeight(w, gnb),
        .q = try QLinear.load(w, a, qp),
        .k = try QLinear.load(w, a, kp),
        .v = try QLinear.load(w, a, vp),
        .o = try QLinear.load(w, a, op),
    };
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: Vae = undefined;
    v.allocator = allocator;
    v.s = s;
    v.conv_in_w = try ownWeight(&w, "decoder.conv_in.conv2d.weight");
    v.conv_in_b = try ownWeight(&w, "decoder.conv_in.conv2d.bias");
    v.mid_r0 = try loadResnet(&w, allocator, "decoder.mid_block.resnets.0");
    v.mid_attn = try loadAttn(&w, allocator, "decoder.mid_block.attentions.0");
    v.mid_r1 = try loadResnet(&w, allocator, "decoder.mid_block.resnets.1");
    for (0..4) |bi| {
        for (0..3) |ri| {
            const pfx = try std.fmt.allocPrint(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            v.up_resnets[bi][ri] = try loadResnet(&w, allocator, pfx);
        }
    }
    for (0..3) |bi| {
        const wk = try std.fmt.allocPrint(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi});
        defer allocator.free(wk);
        const bk = try std.fmt.allocPrint(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi});
        defer allocator.free(bk);
        v.up_conv_w[bi] = try ownWeight(&w, wk);
        v.up_conv_b[bi] = try ownWeight(&w, bk);
    }
    v.norm_out_w = try ownWeight(&w, "decoder.conv_norm_out.norm.weight");
    v.norm_out_b = try ownWeight(&w, "decoder.conv_norm_out.norm.bias");
    v.conv_out_w = try ownWeight(&w, "decoder.conv_out.conv2d.weight");
    v.conv_out_b = try ownWeight(&w, "decoder.conv_out.conv2d.bias");
    return v;
}
