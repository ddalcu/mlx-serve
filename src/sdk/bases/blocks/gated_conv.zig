//! LFM2's gated short convolution: `out(C * conv(B * x))` over the slot's conv window, one fused dispatch at a
//! decode step once the window exists (a spec verify keeps the composed ops, which leave the conv input).
//! copied_from: plugins/lfm2/gated_conv.zig (`gatedConv`).

const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const Linear = @import("linear.zig").Linear;
const conv_window = @import("conv_window.zig");
const fusedGatedConvStep = @import("../../blocks/conv/gated_conv_step.zig").fusedGatedConvStep;

const Array = ops.Array;
const Ctx = block.Ctx;

pub const GatedConv = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        /// `[3 * hidden]` out: B | C | x.
        in_proj: Linear,
        /// Depthwise taps `[hidden, kernel, 1]`.
        conv: Array,
        out_proj: Linear,
        hidden: c_int,
        kernel: c_int,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return error.BatchedRowsUnsupported;
        const g = c.g;
        const h = w.hidden;
        const proj = try w.in_proj.forward(g, x);
        if (c.rows == 1 and !c.capture_spec) if (try kv.state(c, .conv)) |state| {
            if (try fusedGatedConvStep(g.s, proj, state, w.conv, h, w.kernel)) |r| {
                const gated = g.adopt(r.gated) catch |e| {
                    _ = @import("../../../mlx/ffi.zig").mlx_array_free(r.state);
                    return e;
                };
                try kv.setState(c, .conv, try g.adopt(r.state));
                return w.out_proj.forward(g, gated);
            }
        };
        const b_gate = try g.sliceAxis(proj, 2, 0, h);
        const c_gate = try g.sliceAxis(proj, 2, h, 2 * h);
        const x_conv = try g.sliceAxis(proj, 2, 2 * h, 3 * h);
        const conv_out = try conv_window.conv(c, try g.mul(b_gate, x_conv), w.conv, .{}, h, w.kernel, false);
        return w.out_proj.forward(g, try g.mul(c_gate, conv_out));
    }
};
