//! The feed-forward blocks, in the legacy forwards' exact op order. copied_from: sdk/blocks/mlp/dense.zig
//! (`standardMlp`, `denseMLP`, `simpleMLP`), sdk/blocks/mlp/activations.zig (`computeGeglu` silu arm,
//! `reluSquared`, `geluUncompiled`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const linear = @import("linear.zig");
const row_exact = @import("row_exact.zig");
const fused_act = @import("../../blocks/mlp/fused_activations.zig");
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;

/// `silu(gate) * up`: one exact fused dispatch where it serves, else the two ops.
pub fn swiglu(g: *Graph, gate: Array, up: Array) !Array {
    if (try fused_act.fusedSwiGLU(g.s, gate, up)) |y| return g.adopt(y);
    var t = g.scope();
    defer t.deinit();
    return g.keep(try t.mul(try t.mul(gate, try t.sigmoid(gate)), up));
}

fn rowsOf(x: Array) c_int {
    const sh = ops.shape(x);
    var n: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| n *= d;
    return n;
}

/// The dense decoder's SwiGLU (`standardMlp`): gate|up joined only on a solo one-row step of a layer whose q|k|v
/// group was built too (the legacy ties both groups to one flag).
pub const SwiGlu = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        gate: Linear,
        up: Linear,
        down: Linear,
        gu: Joined = .{},
        /// The layer's q|k|v group is built: one-row steps join gate|up.
        rows_joined: bool = false,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const joined = c.mode != .batched_rows and w.rows_joined and w.gu.built() and c.rows * c.slots <= FUSED_ROWS_MAX_M;
        var gate: Array = undefined;
        var up: Array = undefined;
        if (joined) {
            const both = try w.gu.lin.forward(g, x);
            gate = try w.gu.part(g, both, 0);
            up = try w.gu.part(g, both, 1);
        } else {
            gate = try w.gate.forward(g, x);
            up = try w.up.forward(g, x);
        }
        return w.down.forward(g, try swiglu(g, gate, up));
    }
};

/// The layered decoder's and LFM2's SwiGLU (`denseMLP`): gate|up joined at one row, one fused dispatch over the
/// joined halves when they are equal.
pub const GatedMlp = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        gate: Linear,
        up: Linear,
        down: Linear,
        gu: Joined = .{},
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (try c.offloadMlp(x)) |y| return y;
        const g = c.g;
        const lane = c.row_exact;
        if (w.gu.built() and row_exact.groupServes(lane, rowsOf(x))) {
            const both = try row_exact.linear(lane, g, &w.gu.lin, x);
            if (w.gu.widths[0] == w.gu.widths[1]) {
                if (try fused_act.swigluJoined(g.s, both)) |act| return row_exact.linear(lane, g, &w.down, try g.adopt(act));
            }
            return row_exact.linear(lane, g, &w.down, try swiglu(g, try w.gu.part(g, both, 0), try w.gu.part(g, both, 1)));
        }
        return row_exact.linear(lane, g, &w.down, try swiglu(g, try row_exact.linear(lane, g, &w.gate, x), try row_exact.linear(lane, g, &w.up, x)));
    }
};

/// Nemotron-H's ungated MLP: `down(relu(up(x))^2)`.
pub const ReluSq = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct { up: Linear, down: Linear };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const up = try row_exact.linear(c.row_exact, g, &w.up, x);
        const relu = try g.maximum(up, try g.scalar(0, .bfloat16));
        return row_exact.linear(c.row_exact, g, &w.down, try g.square(relu));
    }
};

/// BERT's MLP: `out(gelu_tanh(inter(x)))`, the tanh GELU as its eight bf16 ops.
pub const GeluMlp = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct { inter: Linear, out: Linear };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        return w.out.forward(g, try geluTanh(g, try w.inter.forward(g, x)));
    }
};

/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))` with bf16 constants, in the legacy order.
pub fn geluTanh(g: *Graph, x: Array) !Array {
    var c = g.scope();
    defer c.deinit();
    const x3 = try c.call(mlx.mlx_power, .{ x, try c.scalar(3.0, .bfloat16) });
    const sum = try c.add(x, try c.mul(try c.scalar(0.044715, .bfloat16), x3));
    const t = try c.tanh(try c.mul(try c.scalar(0.7978845608028654, .bfloat16), sum));
    const x_times = try c.mul(x, try c.add(try c.scalar(1.0, .bfloat16), t));
    return g.keep(try c.mul(x_times, try c.scalar(0.5, .bfloat16)));
}
