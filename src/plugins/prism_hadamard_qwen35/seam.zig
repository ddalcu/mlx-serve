//! A Prism pack's norms and residual seams. Where the kernel serves (block 1024, up to 16 rows) a seam adds the
//! residual, takes the next block's norm and rotates it in one dispatch (`rht.normRotate`), seeding the registry's
//! memo so the rotated projections that read it skip their own rotation; elsewhere it is the layered decoder's seam.
//! copied_from: engines/mlx/forward/layered.zig (the Qwen3.5 seams), sdk/blocks/norms/rms.zig (`rhtAddNormRotate`).

const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const seam = @import("../../sdk/bases/blocks/seam.zig");
const norm = @import("../../sdk/bases/blocks/norm.zig");
const rht = @import("../../sdk/quant/hadamard/rht.zig");

const Array = ops.Array;
const Ctx = block.Ctx;

/// RMS norm over the last axis, with what its reader needs of the fused seam.
pub const RotNorm = struct {
    pub const kind: block.Kind = .norm;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        w: Array,
        /// The registry whose memo the fused seam seeds; null = the plain seam.
        reg: ?*rht.Registry = null,
        /// The reader also reads the norm unrotated (a GDN layer's dense projections, the head's hidden): the seam
        /// returns the norm with its rotation seeded; else the rotation alone, which only rotated projections read.
        want_normed: bool = true,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return norm.RmsNorm.forward(&w.w, c, x);
    }
};

/// `RotNorm`'s seam: the fused add + norm + rotation, else `AddNorm`.
pub const RotAddNorm = struct {
    pub const kind: block.Kind = .seam;
    pub const supports_batched_rows = true;
    pub const Weights = void;

    fn fused(c: *Ctx, h: Array, delta: Array, nw: *const RotNorm.Weights) !?block.Mid {
        const reg = nw.reg orelse return null;
        const sh = ops.shape(h);
        const signs = reg.signsFor(sh[sh.len - 1]) orelse return null;
        const r = (try rht.normRotate(h, delta, nw.w, c.shared.eps_arr, signs, reg.block, nw.want_normed, c.stream())) orelse return null;
        const sum = try c.g.adopt(r.sum.?);
        const key = try c.g.adopt(r.normed orelse r.rot);
        if (r.normed != null) _ = try c.g.adopt(r.rot);
        try reg.seed(key, signs, r.rot);
        return .{ .h = sum, .normed = key };
    }

    pub fn mid(c: *Ctx, h: Array, delta: Array, pre: Array, comptime Norm: type, nw: *const Norm.Weights) !block.Mid {
        if (pre.ctx == null) if (try fused(c, h, delta, nw)) |m| return m;
        return seam.AddNorm.mid(c, h, delta, pre, norm.RmsNorm, &nw.w);
    }

    pub fn end(c: *Ctx, h: Array, delta: Array, pre: Array, comptime Norm: type, next: ?*const Norm.Weights) !block.End {
        const nw = next orelse return seam.AddNorm.end(c, h, delta, pre, norm.RmsNorm, null);
        if (pre.ctx == null) if (try fused(c, h, delta, nw)) |m| return .{ .h = m.h, .normed = m.normed };
        return seam.AddNorm.end(c, h, delta, pre, norm.RmsNorm, &nw.w);
    }
};
