//! The Mamba2 mixer (Nemotron-H): in-projection, then conv + SSM + SiLU gate as one fused dispatch over any window
//! the kernel serves (the op chain otherwise), the grouped RMS norm, the out-projection. Conv window and f32 scan
//! state live in the slot's cache (`CacheOps.state`); a spec verify also leaves the conv input and the state after
//! every row. copied_from: sdk/blocks/ssm/mamba2.zig (`mamba2Mixer`, `mamba2GroupNorm`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const Linear = @import("linear.zig").Linear;
const row_exact = @import("row_exact.zig");
const conv_window = @import("conv_window.zig");
const mamba2_decode = @import("../../blocks/ssm/mamba2_decode.zig");
const RECURRENCE_EVAL_INTERVAL = @import("../run.zig").RECURRENCE_EVAL_INTERVAL;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

/// Widest window the fused Mamba2 step serves; tests lower it to force the op chain.
pub var fused_max_rows: c_int = std.math.maxInt(c_int);

pub const Mamba2 = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;

    pub const Geo = struct {
        heads: c_int,
        head_dim: c_int,
        groups: c_int,
        state: c_int,
        kernel: c_int,
        /// The softplus(dt) clamp (`time_step_limit`).
        dt_min: f32,
        dt_max: f32,

        fn inner(geo: Geo) c_int {
            return geo.heads * geo.head_dim;
        }
        fn convDim(geo: Geo) c_int {
            return geo.inner() + 2 * geo.groups * geo.state;
        }
    };

    pub const Weights = struct {
        /// `[inner + conv_dim + heads]` out: z | x B C | dt.
        in_proj: Linear,
        conv_w: Array,
        /// Null = no conv bias (the fused step needs one).
        conv_b: Array = .{},
        A_log: Array,
        D: Array,
        dt_bias: Array,
        norm: Array,
        out_proj: Linear,
        geo: Geo,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return error.BatchedRowsUnsupported;
        const g = c.g;
        const proj = try row_exact.linear(c.row_exact, g, &w.in_proj, x);
        // The test seam the legacy reads, so both arms are reachable from one proof.
        const y = if (c.rows <= fused_max_rows and w.conv_b.ctx != null)
            (try fused(w, c, proj)) orelse try chain(w, c, proj)
        else
            try chain(w, c, proj);
        return row_exact.linear(c.row_exact, g, &w.out_proj, try groupNorm(w, c, y));
    }

    /// The gated `y` from the one-dispatch step, or null where the kernel declines.
    fn fused(w: *const Weights, c: *Ctx, proj: Array) !?Array {
        const g = c.g;
        const geo = w.geo;
        const conv_in = (try kv.state(c, .conv)) orelse try g.zeros(&.{ 1, geo.kernel - 1, geo.convDim() }, ops.dtype(proj));
        const scan = (try kv.state(c, .ssm)) orelse try g.zeros(&.{ 1, geo.heads, geo.head_dim, geo.state }, .float32);
        const out = (try mamba2_decode.step(
            .{ .batch = 1, .rows = c.rows, .heads = geo.heads, .groups = geo.groups, .head_dim = geo.head_dim, .state = geo.state, .conv_kernel = geo.kernel, .proj = ops.shape(proj)[2] },
            .{ .proj = proj, .conv_state = conv_in, .ssm_state = scan, .conv_w = w.conv_w, .conv_b = w.conv_b, .A_log = w.A_log, .dt_bias = w.dt_bias, .D = w.D, .dt_lo = try g.scalar(geo.dt_min, .float32), .dt_hi = try g.scalar(geo.dt_max, .float32), .rows = try g.adopt(mlx.mlx_array_new_int(c.rows)), .all_rows = c.capture_spec },
            g.s,
        )) orelse return null;
        const y = g.adopt(out.y) catch |e| {
            _ = mlx.mlx_array_free(out.ssm_state);
            _ = mlx.mlx_array_free(out.conv_state);
            return e;
        };
        const states = g.adopt(out.ssm_state) catch |e| {
            _ = mlx.mlx_array_free(out.conv_state);
            return e;
        };
        const window = try g.adopt(out.conv_state);
        if (c.capture_spec) {
            const x_conv = try g.sliceAxis(proj, 2, geo.inner(), geo.inner() + geo.convDim());
            try kv.setState(c, .conv_input, try g.concat(&.{ conv_in, x_conv }, 1));
            try kv.setState(c, .state_seq, states);
        }
        const srows = ops.shape(states)[0];
        const last = try g.reshape(try g.sliceAxis(states, 0, srows - 1, srows), &.{ 1, geo.heads, geo.head_dim, geo.state });
        try kv.setState(c, .ssm, last);
        try kv.setState(c, .conv, window);
        return y;
    }

    /// The op chain: conv + SiLU, then the scan one row at a time in f32, `y` back in the input dtype, SiLU(z) gate.
    fn chain(w: *const Weights, c: *Ctx, proj: Array) !Array {
        const g = c.g;
        const geo = w.geo;
        const s = c.rows;
        const inner = geo.inner();
        const cd = geo.convDim();
        const gate = try g.sliceAxis(proj, 2, 0, inner);
        const conv_in = try g.sliceAxis(proj, 2, inner, inner + cd);
        const dt_raw = try g.sliceAxis(proj, 2, inner + cd, inner + cd + geo.heads);

        const conv_out = try conv_window.conv(c, conv_in, w.conv_w, w.conv_b, cd, geo.kernel, true);
        const b_end = inner + geo.groups * geo.state;
        const x_h = try g.reshape(try g.sliceAxis(conv_out, 2, 0, inner), &.{ 1, s, geo.heads, geo.head_dim });
        const B_h = try g.reshape(try g.sliceAxis(conv_out, 2, inner, b_end), &.{ 1, s, geo.groups, geo.state });
        const C_h = try g.reshape(try g.sliceAxis(conv_out, 2, b_end, cd), &.{ 1, s, geo.groups, geo.state });

        var scan = (try kv.state(c, .ssm)) orelse try g.zeros(&.{ 1, geo.heads, geo.head_dim, geo.state }, .float32);

        const dt_sp = try g.call(mlx.mlx_log1p, .{try g.exp(try g.add(try g.astype(dt_raw, .float32), w.dt_bias))});
        const dt = try g.minimum(try g.maximum(dt_sp, try g.scalar(geo.dt_min, .float32)), try g.scalar(geo.dt_max, .float32));
        // A = -exp(A_log) in f32, as the reference casts it to dt's dtype.
        const A = try g.astype(try g.neg(try g.exp(w.A_log)), .float32);
        const D = try g.expandDims(w.D, 1);

        const repeats = @divExact(geo.heads, geo.groups);
        const T: usize = @intCast(s);
        const ys = try g.a.alloc(Array, T);
        defer g.a.free(ys);
        const seq = try g.a.alloc(Array, if (c.capture_spec) T else 0);
        defer g.a.free(seq);
        for (0..T) |t| {
            var tg = g.scope();
            defer tg.deinit();
            const ti: c_int = @intCast(t);
            const dt_t = try tg.reshape(try tg.sliceAxis(dt, 1, ti, ti + 1), &.{ 1, geo.heads });
            const x_t = try tg.reshape(try tg.sliceAxis(x_h, 1, ti, ti + 1), &.{ 1, geo.heads, geo.head_dim });
            const B_t = try groupRows(&tg, B_h, ti, geo, repeats);
            const C_t = try groupRows(&tg, C_h, ti, geo, repeats);
            const dA = try tg.expandDims(try tg.expandDims(try tg.exp(try tg.mul(A, dt_t)), 2), 3);
            const decayed = try tg.mul(scan, dA);
            const dtx = try tg.expandDims(try tg.mul(x_t, try tg.expandDims(dt_t, 2)), 3);
            const next = try tg.add(decayed, try tg.mul(dtx, try tg.expandDims(B_t, 2)));
            const y_state = try tg.sum(try tg.mul(next, try tg.expandDims(C_t, 2)), -1, false);
            ys[t] = try g.keep(try tg.add(y_state, try tg.mul(x_t, D)));
            const prev = scan;
            scan = try g.keep(next);
            if (c.capture_spec) seq[t] = scan else if (t > 0) g.drop(prev);
            if ((t + 1) % RECURRENCE_EVAL_INTERVAL == 0) try ops.eval(scan);
        }
        try kv.setState(c, .ssm, scan);
        if (c.capture_spec) try kv.setState(c, .state_seq, try stack(g, seq));

        const stacked = try stack(g, ys);
        const y_f32 = try g.reshape(try g.transpose(stacked, &.{ 1, 0, 2, 3 }), &.{ 1, s, inner });
        const y = try g.astype(y_f32, ops.dtype(x_h));
        return g.mul(try g.mul(gate, try g.sigmoid(gate)), y);
    }

    fn stack(g: *Graph, xs: []const Array) !Array {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        return g.call(mlx.mlx_stack_axis, .{ v, @as(c_int, 0) });
    }

    /// Row `t` of `[1, S, groups, state]`, repeated per head: `[1, heads, state]`.
    fn groupRows(parent: *Graph, x: Array, t: c_int, geo: Geo, repeats: c_int) !Array {
        var sc = parent.scope();
        defer sc.deinit();
        const g = &sc;
        const r = try g.reshape(try g.sliceAxis(x, 1, t, t + 1), &.{ 1, geo.groups, geo.state });
        return parent.keep(if (repeats > 1) try g.repeat(r, repeats, 1) else r);
    }

    /// Grouped RMS norm over `inner / groups` (not head_dim) times the norm weight: the kernel, else the ops.
    fn groupNorm(w: *const Weights, c: *Ctx, y: Array) !Array {
        const g = c.g;
        const geo = w.geo;
        if (try mamba2_decode.groupNorm(y, w.norm, c.shared.eps_arr, geo.groups, g.s)) |out| return g.adopt(out);
        const gs = @divExact(geo.inner(), geo.groups);
        const grouped = try g.reshape(y, &.{ 1, c.rows, geo.groups, gs });
        const normed = try g.rmsNorm(grouped, try g.ones(&.{gs}, ops.dtype(y)), c.shared.eps);
        return g.mul(try g.reshape(normed, &.{ 1, c.rows, geo.inner() }), w.norm);
    }
};
