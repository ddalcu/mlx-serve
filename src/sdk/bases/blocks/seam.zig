//! The residual seams: `h + delta`, and the norm of the result the next block reads, fused where a kernel serves
//! it (bit-equal to the two ops). A norm that is not `fusable` always takes the two ops.
//! copied_from: sdk/blocks/mlp/dense.zig (`residualTail`), sdk/blocks/norms/rms.zig (`narrowAddNorm`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const add_norm = @import("../../blocks/norms/add_norm.zig");
const residual_norm = @import("../../blocks/norms/residual_norm.zig");

const Array = ops.Array;
const Ctx = block.Ctx;
const Mid = block.Mid;
const End = block.End;

/// Two new handles owned by `g` (the second freed if the first cannot be).
fn take2(g: *ops.Graph, a: Array, b: Array) ![2]Array {
    const x = g.adopt(a) catch |e| {
        _ = mlx.mlx_array_free(b);
        return e;
    };
    return .{ x, try g.adopt(b) };
}

fn rawWeight(comptime Norm: type, nw: *const Norm.Weights) ?Array {
    if (!Norm.fusable) return null;
    return nw.*;
}

/// `delta` under a `PostNormed` block's output norm `pre` (null-ctx: none), at the post-norm epsilon.
fn preNormed(c: *Ctx, delta: Array, pre: Array) !Array {
    return if (pre.ctx == null) delta else c.g.rmsNorm(delta, pre, c.shared.post_eps);
}

/// One dispatch at a solo decode step (`fusedResidualNorm`, the block's output norm `pre` included), the composed ops
/// elsewhere: the dense decoder's and LFM2's seams.
pub const ResidualTail = struct {
    pub const kind: block.Kind = .seam;
    pub const supports_batched_rows = true;
    pub const Weights = void;

    fn fused(c: *const Ctx, h: Array, delta: Array, pre: Array, w_post: Array) !?residual_norm.AddNormResult {
        if (c.mode != .decode) return null;
        return residual_norm.fusedResidualNorm(c.stream(), h, delta, .{ .w1 = pre, .eps1 = c.shared.post_eps_arr, .w_post = w_post, .eps = c.shared.eps_arr });
    }

    pub fn mid(c: *Ctx, h: Array, delta: Array, pre: Array, comptime Norm: type, nw: *const Norm.Weights) !Mid {
        if (rawWeight(Norm, nw)) |w| if (try fused(c, h, delta, pre, w)) |r| {
            const t = try take2(c.g, r.sum, r.normed);
            return .{ .h = t[0], .normed = t[1] };
        };
        const sum = try c.g.add(h, try preNormed(c, delta, pre));
        return .{ .h = sum, .normed = try Norm.forward(nw, c, sum) };
    }

    pub fn end(c: *Ctx, h: Array, delta: Array, pre: Array, comptime Norm: type, next: ?*const Norm.Weights) !End {
        const w: Array = if (next) |nw| rawWeight(Norm, nw) orelse .{} else .{};
        if (try fused(c, h, delta, pre, w)) |r| {
            if (r.normed.ctx == null) return .{ .h = try c.g.adopt(r.sum) };
            const t = try take2(c.g, r.sum, r.normed);
            return .{ .h = t[0], .normed = t[1] };
        }
        return .{ .h = try c.g.add(h, try preNormed(c, delta, pre)) };
    }
};

/// `add_norm.addNorm` at widths up to `add_norm_max_rows`: the layered decoder's (Qwen 3.5) and Nemotron-H's seams.
pub const AddNorm = struct {
    pub const kind: block.Kind = .seam;
    pub const supports_batched_rows = true;
    pub const Weights = void;

    fn narrow(c: *const Ctx, h: Array, delta: Array, w: ?Array) !?add_norm.Out {
        const nw = w orelse return null;
        if (nw.ctx == null) return null;
        const sh = ops.shape(h);
        if (sh.len != 3 or sh[0] * sh[1] > residual_norm.add_norm_max_rows) return null;
        return add_norm.addNorm(h, delta, nw, c.shared.eps_arr, c.stream());
    }

    /// Whether `end` folds the next norm in at this width (an MoE block that combines inside the seam asks).
    pub fn fuses(c: *const Ctx, h: Array) bool {
        const sh = ops.shape(h);
        return sh.len == 3 and sh[0] * sh[1] <= residual_norm.add_norm_max_rows and add_norm.eligible(sh[2], ops.dtype(h), c.stream());
    }

    pub fn mid(c: *Ctx, h: Array, delta_raw: Array, pre: Array, comptime Norm: type, nw: *const Norm.Weights) !Mid {
        const delta = try preNormed(c, delta_raw, pre);
        const w = rawWeight(Norm, nw);
        if (try narrow(c, h, delta, w)) |r| {
            const t = try take2(c.g, r.h, r.normed);
            return .{ .h = t[0], .normed = t[1] };
        }
        const sum = try c.g.add(h, delta);
        return .{ .h = sum, .normed = try Norm.forward(nw, c, sum) };
    }

    pub fn end(c: *Ctx, h: Array, delta_raw: Array, pre: Array, comptime Norm: type, next: ?*const Norm.Weights) !End {
        const delta = try preNormed(c, delta_raw, pre);
        const w = if (next) |nw| rawWeight(Norm, nw) else null;
        if (try narrow(c, h, delta, w)) |r| {
            const t = try take2(c.g, r.h, r.normed);
            return .{ .h = t[0], .normed = t[1] };
        }
        return .{ .h = try c.g.add(h, delta) };
    }
};
