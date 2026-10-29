//! K2-Horizon's grouped RMS norm (`K2HorizonRMSNorm`): every residual-width norm normalizes `groups` equal channel
//! groups on their own rms, then applies the full-width weight; any other width (a per-head norm) is a plain RMS norm.
//! copied_from: sdk/blocks/norms/grouped_rms.zig (`groupedRmsNorm`), sdk/blocks/norms/rms.zig (`rmsNorm`'s width
//! test). Never fused into a residual seam, as the legacy declined every fused seam for grouped norms.

const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");

const Array = ops.Array;
const Ctx = block.Ctx;

pub const GroupedRmsNorm = struct {
    pub const kind: block.Kind = .norm;
    pub const supports_batched_rows = true;
    pub const fusable = false;
    pub const Weights = struct {
        w: Array,
        groups: c_int,
        /// The residual width the grouping applies to.
        hidden: c_int,
        /// The weight-less rms norm's ones, one group wide, in the activation dtype.
        ones: Array,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const sh = ops.shape(x);
        const width = sh[sh.len - 1];
        if (w.groups <= 1 or width != w.hidden) return g.rmsNorm(x, w.w, c.shared.eps);
        var gs: [8]c_int = undefined;
        @memcpy(gs[0 .. sh.len - 1], sh[0 .. sh.len - 1]);
        gs[sh.len - 1] = w.groups;
        gs[sh.len] = @divExact(width, w.groups);
        var t = g.scope();
        defer t.deinit();
        const normed = try t.rmsNorm(try t.reshape(x, gs[0 .. sh.len + 1]), w.ones, c.shared.eps);
        return g.keep(try t.mul(try t.reshape(normed, sh), w.w));
    }
};
